import json
import time
from typing import Any, AsyncGenerator, AsyncIterable, Dict
from fastapi import HTTPException
from fastapi.encoders import jsonable_encoder
from mlflow.gateway.config import CohereConfig, RouteConfig
from mlflow.gateway.providers.base import BaseProvider, ProviderAdapter
from mlflow.gateway.providers.utils import rename_payload_keys, send_request, send_stream_request
from mlflow.gateway.schemas import chat, completions, embeddings
class CohereAdapter(ProviderAdapter):

    @staticmethod
    def _scale_temperature(payload):
        if (temperature := payload.get('temperature')):
            payload['temperature'] = 2.5 * temperature
        return payload

    @classmethod
    def model_to_completions(cls, resp, config):
        return completions.ResponsePayload(created=int(time.time()), object='text_completion', model=config.model.name, choices=[completions.Choice(index=idx, text=c['text'], finish_reason=None) for idx, c in enumerate(resp['generations'])], usage=completions.CompletionsUsage(prompt_tokens=None, completion_tokens=None, total_tokens=None))

    @classmethod
    def model_to_completions_streaming(cls, resp, config):
        response = resp.get('response')
        return completions.StreamResponsePayload(id=response['id'] if response else None, created=int(time.time()), model=config.model.name, choices=[completions.StreamChoice(index=resp.get('index', 0), finish_reason=resp.get('finish_reason'), delta=completions.StreamDelta(role=None, content=resp.get('text')))], usage=completions.CompletionsUsage(prompt_tokens=None, completion_tokens=None, total_tokens=None))

    @classmethod
    def model_to_embeddings(cls, resp, config):
        return embeddings.ResponsePayload(data=[embeddings.EmbeddingObject(embedding=output, index=idx) for idx, output in enumerate(resp['embeddings'])], model=config.model.name, usage=embeddings.EmbeddingsUsage(prompt_tokens=None, total_tokens=None))

    @classmethod
    def completions_to_model(cls, payload, config):
        key_mapping = {'stop': 'stop_sequences', 'n': 'num_generations'}
        cls.check_keys_against_mapping(key_mapping, payload)
        payload = cls._scale_temperature(payload)
        return rename_payload_keys(payload, key_mapping)

    @classmethod
    def completions_streaming_to_model(cls, payload, config):
        return cls.completions_to_model(payload, config)

    @classmethod
    def embeddings_to_model(cls, payload, config):
        key_mapping = {'input': 'texts'}
        for k1, k2 in key_mapping.items():
            if k2 in payload:
                raise HTTPException(status_code=422, detail=f'Invalid parameter {k2}. Use {k1} instead.')
        return rename_payload_keys(payload, key_mapping)

    @classmethod
    def chat_to_model(cls, payload, config):
        if payload['n'] != 1:
            raise HTTPException(status_code=422, detail=f'Parameter n must be 1 for Cohere chat, got {payload['n']}.')
        del payload['n']
        if 'stop' in payload:
            raise HTTPException(status_code=422, detail='Parameter stop is not supported for Cohere chat.')
        payload = cls._scale_temperature(payload)
        messages = payload.pop('messages')
        last_message = messages.pop()
        if last_message['role'] != 'user':
            raise HTTPException(status_code=422, detail=f'Last message must be from user, got {last_message['role']}.')
        payload['message'] = last_message['content']
        system_messages = [m for m in messages if m['role'] == 'system']
        if len(system_messages) > 0:
            payload['preamble_override'] = '\n'.join((m['content'] for m in system_messages))
        messages = [m for m in messages if m['role'] in ('user', 'assistant')]
        if messages:
            payload['chat_history'] = [{'role': 'USER' if m['role'] == 'user' else 'CHATBOT', 'message': m['content']} for m in messages]
        return payload

    @classmethod
    def chat_streaming_to_model(cls, payload, config):
        return cls.chat_to_model(payload, config)

    @classmethod
    def model_to_chat(cls, resp, config):
        return chat.ResponsePayload(id=resp['response_id'], object='chat.completion', created=int(time.time()), model=config.model.name, choices=[chat.Choice(index=0, message=chat.ResponseMessage(role='assistant', content=resp['text']), finish_reason=None)], usage=chat.ChatUsage(prompt_tokens=resp['token_count']['prompt_tokens'], completion_tokens=resp['token_count']['response_tokens'], total_tokens=resp['token_count']['total_tokens']))

    @classmethod
    def model_to_chat_streaming(cls, resp, config):
        response = resp.get('response')
        return chat.StreamResponsePayload(id=response['response_id'] if response else None, created=int(time.time()), model=config.model.name, choices=[chat.StreamChoice(index=0, finish_reason=resp.get('finish_reason'), delta=chat.StreamDelta(role=None, content=resp.get('text')))], usage=chat.ChatUsage(prompt_tokens=response['token_count']['prompt_tokens'] if response else None, completion_tokens=response['token_count']['response_tokens'] if response else None, total_tokens=response['token_count']['total_tokens'] if response else None))