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
def _process_embeddings_response_for_mlflow_serving(self, response):
    try:
        validated_response = EmbeddingsResponse(**response)
        inference_data = validated_response.predictions
    except ValidationError as e:
        raise HTTPException(status_code=502, detail=str(e))
    return inference_data