from typing import Any, Dict, List, Optional
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel, Extra, root_validator
from langchain_community.llms.sagemaker_endpoint import ContentHandlerBase
class EmbeddingsContentHandler(ContentHandlerBase[List[str], List[List[float]]]):
    """Content handler for LLM class."""