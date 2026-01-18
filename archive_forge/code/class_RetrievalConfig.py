from typing import Any, Dict, List, Optional
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import BaseModel, root_validator
from langchain_core.retrievers import BaseRetriever
class RetrievalConfig(BaseModel, extra='allow'):
    """Configuration for retrieval."""
    vectorSearchConfiguration: VectorSearchConfig