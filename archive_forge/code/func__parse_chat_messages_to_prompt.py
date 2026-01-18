import time
from contextlib import contextmanager
from typing import Any, Dict, List
from fastapi import HTTPException
from fastapi.encoders import jsonable_encoder
from mlflow.exceptions import MlflowException
from mlflow.gateway.config import MosaicMLConfig, RouteConfig
from mlflow.gateway.providers.base import BaseProvider
from mlflow.gateway.providers.utils import rename_payload_keys, send_request
from mlflow.gateway.schemas import chat, completions, embeddings
@staticmethod
def _parse_chat_messages_to_prompt(messages: List[chat.RequestMessage]) -> str:
    """
        This parser is based on the format described in
        https://huggingface.co/blog/llama2#how-to-prompt-llama-2 .
        The expected format is:
            "<s>[INST] <<SYS>>
            {{ system_prompt }}
            <</SYS>>

            {{ user_msg_1 }} [/INST] {{ model_answer_1 }} </s>
            <s>[INST] {{ user_msg_2 }} [/INST]"
        """
    prompt = '<s>'
    for m in messages:
        if m.role == 'system' or m.role == 'user':
            inst = m.content
            if m.role == 'system':
                inst = f'<<SYS>> {inst} <</SYS>>'
            inst += ' [/INST]'
            if prompt.endswith('[/INST]'):
                prompt = prompt[:-7]
            else:
                inst = f'[INST] {inst}'
            prompt += inst
        elif m.role == 'assistant':
            prompt += f' {m.content} </s><s>'
        else:
            raise MlflowException.invalid_parameter_value(f"Invalid role {m.role} inputted. Must be one of 'system', 'user', or 'assistant'.")
    if prompt.endswith('</s><s>'):
        prompt = prompt[:-7]
    return prompt