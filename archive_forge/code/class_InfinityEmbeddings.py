import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Tuple
import aiohttp
import numpy as np
import requests
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel, Extra, root_validator
from langchain_core.utils import get_from_dict_or_env
class InfinityEmbeddings(BaseModel, Embeddings):
    """Embedding models for self-hosted https://github.com/michaelfeil/infinity
    This should also work for text-embeddings-inference and other
    self-hosted openai-compatible servers.

    Infinity is a class to interact with Embedding Models on https://github.com/michaelfeil/infinity


    Example:
        .. code-block:: python

            from langchain_community.embeddings import InfinityEmbeddings
            InfinityEmbeddings(
                model="BAAI/bge-small",
                infinity_api_url="http://localhost:7997",
            )
    """
    model: str
    'Underlying Infinity model id.'
    infinity_api_url: str = 'http://localhost:7997'
    'Endpoint URL to use.'
    client: Any = None
    'Infinity client.'

    class Config:
        """Configuration for this pydantic object."""
        extra = Extra.forbid

    @root_validator(allow_reuse=True)
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        values['infinity_api_url'] = get_from_dict_or_env(values, 'infinity_api_url', 'INFINITY_API_URL')
        values['client'] = TinyAsyncOpenAIInfinityEmbeddingClient(host=values['infinity_api_url'])
        return values

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Call out to Infinity's embedding endpoint.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        embeddings = self.client.embed(model=self.model, texts=texts)
        return embeddings

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Async call out to Infinity's embedding endpoint.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        embeddings = await self.client.aembed(model=self.model, texts=texts)
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Call out to Infinity's embedding endpoint.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        return self.embed_documents([text])[0]

    async def aembed_query(self, text: str) -> List[float]:
        """Async call out to Infinity's embedding endpoint.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        embeddings = await self.aembed_documents([text])
        return embeddings[0]