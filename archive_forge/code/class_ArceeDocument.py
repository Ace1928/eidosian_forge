from enum import Enum
from typing import Any, Dict, List, Literal, Mapping, Optional, Union
import requests
from langchain_core.pydantic_v1 import BaseModel, SecretStr, root_validator
from langchain_core.retrievers import Document
class ArceeDocument(BaseModel):
    """Arcee document."""
    index: str
    id: str
    score: float
    source: ArceeDocumentSource