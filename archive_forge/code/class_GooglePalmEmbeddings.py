from __future__ import annotations
import logging
from typing import Any, Callable, Dict, List, Optional
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel, root_validator
from langchain_core.utils import get_from_dict_or_env
from tenacity import (
class GooglePalmEmbeddings(BaseModel, Embeddings):
    """Google's PaLM Embeddings APIs."""
    client: Any
    google_api_key: Optional[str]
    model_name: str = 'models/embedding-gecko-001'
    'Model name to use.'
    show_progress_bar: bool = False
    'Whether to show a tqdm progress bar. Must have `tqdm` installed.'

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate api key, python package exists."""
        google_api_key = get_from_dict_or_env(values, 'google_api_key', 'GOOGLE_API_KEY')
        try:
            import google.generativeai as genai
            genai.configure(api_key=google_api_key)
        except ImportError:
            raise ImportError('Could not import google.generativeai python package.')
        values['client'] = genai
        return values

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if self.show_progress_bar:
            try:
                from tqdm import tqdm
                iter_ = tqdm(texts, desc='GooglePalmEmbeddings')
            except ImportError:
                logger.warning('Unable to show progress bar because tqdm could not be imported. Please install with `pip install tqdm`.')
                iter_ = texts
        else:
            iter_ = texts
        return [self.embed_query(text) for text in iter_]

    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        embedding = embed_with_retry(self, self.model_name, text)
        return embedding['embedding']