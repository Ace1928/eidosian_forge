from __future__ import annotations
import logging
import time
from abc import ABC, abstractmethod
from typing import Callable
import openai
from typing_extensions import override
class OctoAI(LLM):
    """Accessing OctoAI"""

    def __init__(self, model: str, api_key: str) -> None:
        super().__init__(model, api_key)
        self.client = openai.OpenAI(base_url='https://text.octoai.run/v1', api_key=api_key)

    @override
    def query(self, prompt: str) -> str:
        level = logging.getLogger().level
        logging.getLogger().setLevel(logging.WARNING)
        response = self.client.chat.completions.create(model=self.model, messages=[{'role': 'system', 'content': 'You are a helpful assistant. Keep your responses limited to one short paragraph if possible.'}, {'role': 'user', 'content': prompt}], max_tokens=MAX_TOKENS, temperature=TEMPERATURE, top_p=TOP_P)
        logging.getLogger().setLevel(level)
        return response.choices[0].message.content

    @override
    def valid_models(self) -> list[str]:
        return ['llamaguard-7b', 'llama-2-13b-chat']