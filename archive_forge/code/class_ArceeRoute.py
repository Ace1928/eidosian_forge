from enum import Enum
from typing import Any, Dict, List, Literal, Mapping, Optional, Union
import requests
from langchain_core.pydantic_v1 import BaseModel, SecretStr, root_validator
from langchain_core.retrievers import Document
class ArceeRoute(str, Enum):
    """Routes available for the Arcee API as enumerator."""
    generate = 'models/generate'
    retrieve = 'models/retrieve'
    model_training_status = 'models/status/{id_or_name}'