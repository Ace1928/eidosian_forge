from enum import Enum
from typing import Any, Dict, List, Literal, Mapping, Optional, Union
import requests
from langchain_core.pydantic_v1 import BaseModel, SecretStr, root_validator
from langchain_core.retrievers import Document
class DALMFilterType(str, Enum):
    """Filter types available for a DALM retrieval as enumerator."""
    fuzzy_search = 'fuzzy_search'
    strict_search = 'strict_search'