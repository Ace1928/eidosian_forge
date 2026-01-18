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
class SearchRoutesResponse(BaseModel):
    routes: List[Route]
    next_page_token: Optional[str] = None

    class Config:
        schema_extra = {'example': {'routes': [{'name': 'openai-chat', 'route_type': 'llm/v1/chat', 'model': {'name': 'gpt-3.5-turbo', 'provider': 'openai'}}, {'name': 'anthropic-completions', 'route_type': 'llm/v1/completions', 'model': {'name': 'claude-instant-100k', 'provider': 'anthropic'}}, {'name': 'cohere-embeddings', 'route_type': 'llm/v1/embeddings', 'model': {'name': 'embed-english-v2.0', 'provider': 'cohere'}}], 'next_page_token': 'eyJpbmRleCI6IDExfQ=='}}