from typing import List, Literal, Optional
from pydantic import Field
from mlflow.gateway.base_models import RequestModel, ResponseModel
from mlflow.gateway.config import IS_PYDANTIC_V2
class RequestPayload(BaseRequestPayload):
    messages: List[RequestMessage] = Field(..., min_items=1)

    class Config:
        if IS_PYDANTIC_V2:
            json_schema_extra = _REQUEST_PAYLOAD_EXTRA_SCHEMA
        else:
            schema_extra = _REQUEST_PAYLOAD_EXTRA_SCHEMA