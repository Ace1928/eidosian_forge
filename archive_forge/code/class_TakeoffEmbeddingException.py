from enum import Enum
from typing import Any, List, Optional, Set, Union
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel
class TakeoffEmbeddingException(Exception):
    """Exceptions experienced with interfacing with Takeoff Embedding Wrapper"""