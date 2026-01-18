from enum import Enum
from typing import Any, Dict, List, Literal, Mapping, Optional, Union
import requests
from langchain_core.pydantic_v1 import BaseModel, SecretStr, root_validator
from langchain_core.retrievers import Document
class ArceeDocumentSource(BaseModel):
    """Source of an Arcee document."""
    document: str
    name: str
    id: str