from typing import List, Literal, Optional, Union
from mlflow.gateway.base_models import RequestModel, ResponseModel
from mlflow.gateway.config import IS_PYDANTIC_V2
class EmbeddingsUsage(ResponseModel):
    prompt_tokens: Optional[int] = None
    total_tokens: Optional[int] = None