from typing import Any, Dict, List, Optional
import requests
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel, Extra, Field, SecretStr
class HuggingFaceInferenceAPIEmbeddings(BaseModel, Embeddings):
    """Embed texts using the HuggingFace API.

    Requires a HuggingFace Inference API key and a model name.
    """
    api_key: SecretStr
    'Your API key for the HuggingFace Inference API.'
    model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'
    'The name of the model to use for text embeddings.'
    api_url: Optional[str] = None
    'Custom inference endpoint url. None for using default public url.'

    @property
    def _api_url(self) -> str:
        return self.api_url or self._default_api_url

    @property
    def _default_api_url(self) -> str:
        return f'https://api-inference.huggingface.co/pipeline/feature-extraction/{self.model_name}'

    @property
    def _headers(self) -> dict:
        return {'Authorization': f'Bearer {self.api_key.get_secret_value()}'}

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Get the embeddings for a list of texts.

        Args:
            texts (Documents): A list of texts to get embeddings for.

        Returns:
            Embedded texts as List[List[float]], where each inner List[float]
                corresponds to a single input text.

        Example:
            .. code-block:: python

                from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings

                hf_embeddings = HuggingFaceInferenceAPIEmbeddings(
                    api_key="your_api_key",
                    model_name="sentence-transformers/all-MiniLM-l6-v2"
                )
                texts = ["Hello, world!", "How are you?"]
                hf_embeddings.embed_documents(texts)
        """
        response = requests.post(self._api_url, headers=self._headers, json={'inputs': texts, 'options': {'wait_for_model': True, 'use_cache': True}})
        return response.json()

    def embed_query(self, text: str) -> List[float]:
        """Compute query embeddings using a HuggingFace transformer model.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        return self.embed_documents([text])[0]