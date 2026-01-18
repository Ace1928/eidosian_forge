from __future__ import annotations
import logging
from typing import (
from typing_extensions import TypedDict
def _get_openai_encoder() -> Callable[[Sequence[str]], Sequence[Sequence[float]]]:
    """Get the OpenAI GPT-3 encoder."""
    try:
        from openai import Client
    except ImportError:
        raise ImportError('THe default encoder for the EmbeddingDistance class uses the OpenAI API. Please either install the openai library with `pip install openai` or provide a custom encoder function (Callable[[str], Sequence[float]]).')

    def encode_text(texts: Sequence[str]) -> Sequence[Sequence[float]]:
        client = Client()
        response = client.embeddings.create(input=list(texts), model='text-embedding-3-small')
        return [d.embedding for d in response.data]
    return encode_text