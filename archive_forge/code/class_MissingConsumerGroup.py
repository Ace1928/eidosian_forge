from enum import Enum
from typing import Any, List, Optional, Set, Union
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel
class MissingConsumerGroup(TakeoffEmbeddingException):
    """Exception raised when no consumer group is provided on initialization of
    TitanTakeoffEmbed or in embed request"""