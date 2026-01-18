import json
from typing import AsyncIterable
from mlflow.exceptions import MlflowException
from mlflow.gateway.config import OpenAIAPIType, OpenAIConfig, RouteConfig
from mlflow.gateway.providers.base import BaseProvider
from mlflow.gateway.providers.utils import send_request, send_stream_request
from mlflow.gateway.schemas import chat, completions, embeddings
from mlflow.gateway.utils import handle_incomplete_chunks, strip_sse_prefix
from mlflow.utils.uri import append_to_uri_path, append_to_uri_query_params
class OpenAIProvider(BaseProvider):
    NAME = 'OpenAI'

    def __init__(self, config: RouteConfig) -> None:
        super().__init__(config)
        if config.model.config is None or not isinstance(config.model.config, OpenAIConfig):
            raise MlflowException.invalid_parameter_value('Invalid config type {config.model.config}')
        self.openai_config: OpenAIConfig = config.model.config

    @property
    def _request_base_url(self):
        api_type = self.openai_config.openai_api_type
        if api_type == OpenAIAPIType.OPENAI:
            base_url = self.openai_config.openai_api_base or 'https://api.openai.com/v1'
            if (api_version := (self.openai_config.openai_api_version is not None)):
                return append_to_uri_query_params(base_url, ('api-version', api_version))
            else:
                return base_url
        elif api_type in (OpenAIAPIType.AZURE, OpenAIAPIType.AZUREAD):
            openai_url = append_to_uri_path(self.openai_config.openai_api_base, 'openai', 'deployments', self.openai_config.openai_deployment_name)
            return append_to_uri_query_params(openai_url, ('api-version', self.openai_config.openai_api_version))
        else:
            raise MlflowException.invalid_parameter_value(f"Invalid OpenAI API type '{self.openai_config.openai_api_type}'")

    @property
    def _request_headers(self):
        api_type = self.openai_config.openai_api_type
        if api_type == OpenAIAPIType.OPENAI:
            headers = {'Authorization': f'Bearer {self.openai_config.openai_api_key}'}
            if (org := self.openai_config.openai_organization):
                headers['OpenAI-Organization'] = org
            return headers
        elif api_type == OpenAIAPIType.AZUREAD:
            return {'Authorization': f'Bearer {self.openai_config.openai_api_key}'}
        elif api_type == OpenAIAPIType.AZURE:
            return {'api-key': self.openai_config.openai_api_key}
        else:
            raise MlflowException.invalid_parameter_value(f"Invalid OpenAI API type '{self.openai_config.openai_api_type}'")

    def _add_model_to_payload_if_necessary(self, payload):
        if self.openai_config.openai_api_type not in (OpenAIAPIType.AZURE, OpenAIAPIType.AZUREAD):
            return {'model': self.config.model.name, **payload}
        else:
            return payload

    async def chat_stream(self, payload: chat.RequestPayload) -> AsyncIterable[chat.StreamResponsePayload]:
        from fastapi.encoders import jsonable_encoder
        payload = jsonable_encoder(payload, exclude_none=True)
        self.check_for_model_field(payload)
        stream = send_stream_request(headers=self._request_headers, base_url=self._request_base_url, path='chat/completions', payload=self._add_model_to_payload_if_necessary(payload))
        async for chunk in handle_incomplete_chunks(stream):
            chunk = chunk.strip()
            if not chunk:
                continue
            data = strip_sse_prefix(chunk.decode('utf-8'))
            if data == '[DONE]':
                return
            resp = json.loads(data)
            yield chat.StreamResponsePayload(id=resp['id'], object=resp['object'], created=resp['created'], model=resp['model'], choices=[chat.StreamChoice(index=c['index'], finish_reason=c['finish_reason'], delta=chat.StreamDelta(role=c['delta'].get('role'), content=c['delta'].get('content'))) for c in resp['choices']])

    async def chat(self, payload: chat.RequestPayload) -> chat.ResponsePayload:
        from fastapi.encoders import jsonable_encoder
        payload = jsonable_encoder(payload, exclude_none=True)
        self.check_for_model_field(payload)
        resp = await send_request(headers=self._request_headers, base_url=self._request_base_url, path='chat/completions', payload=self._add_model_to_payload_if_necessary(payload))
        return chat.ResponsePayload(id=resp['id'], object=resp['object'], created=resp['created'], model=resp['model'], choices=[chat.Choice(index=idx, message=chat.ResponseMessage(role=c['message']['role'], content=c['message']['content']), finish_reason=c['finish_reason']) for idx, c in enumerate(resp['choices'])], usage=chat.ChatUsage(prompt_tokens=resp['usage']['prompt_tokens'], completion_tokens=resp['usage']['completion_tokens'], total_tokens=resp['usage']['total_tokens']))

    def _prepare_completion_request_payload(self, payload):
        payload['messages'] = [{'role': 'user', 'content': payload.pop('prompt')}]
        return payload

    def _prepare_completion_response_payload(self, resp):
        return completions.ResponsePayload(id=resp['id'], object='text_completion', created=resp['created'], model=resp['model'], choices=[completions.Choice(index=idx, text=c['message']['content'], finish_reason=c['finish_reason']) for idx, c in enumerate(resp['choices'])], usage=completions.CompletionsUsage(prompt_tokens=resp['usage']['prompt_tokens'], completion_tokens=resp['usage']['completion_tokens'], total_tokens=resp['usage']['total_tokens']))

    async def completions_stream(self, payload: completions.RequestPayload) -> AsyncIterable[completions.StreamResponsePayload]:
        from fastapi.encoders import jsonable_encoder
        payload = jsonable_encoder(payload, exclude_none=True)
        self.check_for_model_field(payload)
        payload = self._prepare_completion_request_payload(payload)
        stream = send_stream_request(headers=self._request_headers, base_url=self._request_base_url, path='chat/completions', payload=self._add_model_to_payload_if_necessary(payload))
        async for chunk in handle_incomplete_chunks(stream):
            chunk = chunk.strip()
            if not chunk:
                continue
            data = strip_sse_prefix(chunk.decode('utf-8'))
            if data == '[DONE]':
                return
            resp = json.loads(data)
            yield completions.StreamResponsePayload(id=resp['id'], object='text_completion_chunk', created=resp['created'], model=resp['model'], choices=[completions.StreamChoice(index=c['index'], finish_reason=c['finish_reason'], delta=completions.StreamDelta(content=c['delta'].get('content'))) for c in resp['choices']])

    async def completions(self, payload: completions.RequestPayload) -> completions.ResponsePayload:
        from fastapi.encoders import jsonable_encoder
        payload = jsonable_encoder(payload, exclude_none=True)
        self.check_for_model_field(payload)
        payload = self._prepare_completion_request_payload(payload)
        resp = await send_request(headers=self._request_headers, base_url=self._request_base_url, path='chat/completions', payload=self._add_model_to_payload_if_necessary(payload))
        return self._prepare_completion_response_payload(resp)

    async def embeddings(self, payload: embeddings.RequestPayload) -> embeddings.ResponsePayload:
        from fastapi.encoders import jsonable_encoder
        payload = jsonable_encoder(payload, exclude_none=True)
        self.check_for_model_field(payload)
        resp = await send_request(headers=self._request_headers, base_url=self._request_base_url, path='embeddings', payload=self._add_model_to_payload_if_necessary(payload))
        return embeddings.ResponsePayload(data=[embeddings.EmbeddingObject(embedding=d['embedding'], index=idx) for idx, d in enumerate(resp['data'])], model=resp['model'], usage=embeddings.EmbeddingsUsage(prompt_tokens=resp['usage']['prompt_tokens'], total_tokens=resp['usage']['total_tokens']))