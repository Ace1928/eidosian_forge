from __future__ import annotations
import requests
import json
from typing import Iterator, List, Optional
from ..core.model import ModelInterface
from ..models import ModelConfig
from eidosian_core import eidosian

class OllamaModel(ModelInterface):
    """Adapter for Ollama engines within agent_forge."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.url = "http://127.0.0.1:8938/api/generate"
        self.model_name = config.model_name or "qwen3.5:2b"

    @eidosian()
    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature if temperature is not None else self.config.temperature,
                "num_predict": max_tokens if max_tokens is not None else self.config.max_tokens
            }
        }
        try:
            response = requests.post(self.url, json=payload, timeout=600)
            response.raise_for_status()
            data = response.json()
            return data.get("response", "").strip()
        except Exception as e:
            return f"Error connecting to Ollama: {e}"

    @eidosian()
    def generate_stream(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Iterator[str]:
        response = self.generate(prompt, max_tokens, temperature)
        yield response

    @eidosian()
    def tokenize(self, text: str) -> List[int]:
        return [ord(c) for c in text]

    @eidosian()
    def num_tokens(self, text: str) -> int:
        return len(text)
