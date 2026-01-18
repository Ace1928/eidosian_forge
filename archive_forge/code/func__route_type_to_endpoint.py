from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from fastapi import FastAPI, HTTPException, Request
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from mlflow.deployments.server.config import Endpoint
from mlflow.deployments.server.constants import (
from mlflow.environment_variables import (
from mlflow.exceptions import MlflowException
from mlflow.gateway.base_models import SetLimitsModel
from mlflow.gateway.config import (
from mlflow.gateway.constants import (
from mlflow.gateway.providers import get_provider
from mlflow.gateway.schemas import chat, completions, embeddings
from mlflow.gateway.utils import SearchRoutesToken, make_streaming_response
from mlflow.version import VERSION
def _route_type_to_endpoint(config: RouteConfig, limiter: Limiter, key: str):
    provider_to_factory = {RouteType.LLM_V1_CHAT: _create_chat_endpoint, RouteType.LLM_V1_COMPLETIONS: _create_completions_endpoint, RouteType.LLM_V1_EMBEDDINGS: _create_embeddings_endpoint}
    if (factory := provider_to_factory.get(config.route_type)):
        handler = factory(config)
        if (limit := config.limit):
            limit_value = f'{limit.calls}/{limit.renewal_period}'
            handler.__name__ = f'{handler.__name__}_{config.name}_{key}'
            return limiter.limit(limit_value)(handler)
        else:
            return handler
    raise HTTPException(status_code=404, detail=f'Unexpected route type {config.route_type!r} for route {config.name!r}.')