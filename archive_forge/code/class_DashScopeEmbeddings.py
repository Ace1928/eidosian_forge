from __future__ import annotations
import logging
from typing import (
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel, Extra, root_validator
from langchain_core.utils import get_from_dict_or_env
from requests.exceptions import HTTPError
from tenacity import (
class DashScopeEmbeddings(BaseModel, Embeddings):
    """DashScope embedding models.

    To use, you should have the ``dashscope`` python package installed, and the
    environment variable ``DASHSCOPE_API_KEY`` set with your API key or pass it
    as a named parameter to the constructor.

    Example:
        .. code-block:: python

            from langchain_community.embeddings import DashScopeEmbeddings
            embeddings = DashScopeEmbeddings(dashscope_api_key="my-api-key")

    Example:
        .. code-block:: python

            import os
            os.environ["DASHSCOPE_API_KEY"] = "your DashScope API KEY"

            from langchain_community.embeddings.dashscope import DashScopeEmbeddings
            embeddings = DashScopeEmbeddings(
                model="text-embedding-v1",
            )
            text = "This is a test query."
            query_result = embeddings.embed_query(text)

    """
    client: Any
    'The DashScope client.'
    model: str = 'text-embedding-v1'
    dashscope_api_key: Optional[str] = None
    max_retries: int = 5
    'Maximum number of retries to make when generating.'

    class Config:
        """Configuration for this pydantic object."""
        extra = Extra.forbid

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        import dashscope
        'Validate that api key and python package exists in environment.'
        values['dashscope_api_key'] = get_from_dict_or_env(values, 'dashscope_api_key', 'DASHSCOPE_API_KEY')
        dashscope.api_key = values['dashscope_api_key']
        try:
            import dashscope
            values['client'] = dashscope.TextEmbedding
        except ImportError:
            raise ImportError('Could not import dashscope python package. Please install it with `pip install dashscope`.')
        return values

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Call out to DashScope's embedding endpoint for embedding search docs.

        Args:
            texts: The list of texts to embed.
            chunk_size: The chunk size of embeddings. If None, will use the chunk size
                specified by the class.

        Returns:
            List of embeddings, one for each text.
        """
        embeddings = embed_with_retry(self, input=texts, text_type='document', model=self.model)
        embedding_list = [item['embedding'] for item in embeddings]
        return embedding_list

    def embed_query(self, text: str) -> List[float]:
        """Call out to DashScope's embedding endpoint for embedding query text.

        Args:
            text: The text to embed.

        Returns:
            Embedding for the text.
        """
        embedding = embed_with_retry(self, input=text, text_type='query', model=self.model)[0]['embedding']
        return embedding