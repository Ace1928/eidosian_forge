import time
from typing import List
from fastapi import HTTPException
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, StrictFloat, StrictStr, ValidationError, validator
from mlflow.gateway.config import MlflowModelServingConfig, RouteConfig
from mlflow.gateway.constants import MLFLOW_SERVING_RESPONSE_KEY
from mlflow.gateway.providers.base import BaseProvider
from mlflow.gateway.providers.utils import send_request
from mlflow.gateway.schemas import chat, completions, embeddings
@staticmethod
def _extract_mlflow_response_key(response):
    if MLFLOW_SERVING_RESPONSE_KEY not in response:
        raise HTTPException(status_code=502, detail=f'The response is missing the required key: {MLFLOW_SERVING_RESPONSE_KEY}.')
    return response[MLFLOW_SERVING_RESPONSE_KEY]