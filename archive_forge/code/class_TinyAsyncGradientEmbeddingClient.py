from typing import Any, Dict, List, Optional
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel, Extra, root_validator
from langchain_core.utils import get_from_dict_or_env
from packaging.version import parse
class TinyAsyncGradientEmbeddingClient:
    """Deprecated, TinyAsyncGradientEmbeddingClient was removed.

    This class is just for backwards compatibility with older versions
    of langchain_community.
    It might be entirely removed in the future.
    """

    def __init__(self, *args, **kwargs) -> None:
        raise ValueError('Deprecated,TinyAsyncGradientEmbeddingClient was removed.')