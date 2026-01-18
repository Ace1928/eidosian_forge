from typing import List, Literal, Optional
from pydantic import Field
from mlflow.gateway.base_models import RequestModel, ResponseModel
from mlflow.gateway.config import IS_PYDANTIC_V2
class StreamDelta(ResponseModel):
    role: Optional[str] = None
    content: Optional[str] = None