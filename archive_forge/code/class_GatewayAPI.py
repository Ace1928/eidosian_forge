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
class GatewayAPI(FastAPI):

    def __init__(self, config: GatewayConfig, limiter: Limiter, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.state.limiter = limiter
        self.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
        self.dynamic_routes: Dict[str, Route] = {}
        self.set_dynamic_routes(config, limiter)

    def set_dynamic_routes(self, config: GatewayConfig, limiter: Limiter) -> None:
        self.dynamic_routes.clear()
        for route in config.routes:
            self.add_api_route(path=MLFLOW_DEPLOYMENTS_ENDPOINTS_BASE + route.name + MLFLOW_DEPLOYMENTS_QUERY_SUFFIX, endpoint=_route_type_to_endpoint(route, limiter, 'deployments'), methods=['POST'])
            self.add_api_route(path=f'{MLFLOW_GATEWAY_ROUTE_BASE}{route.name}{MLFLOW_QUERY_SUFFIX}', endpoint=_route_type_to_endpoint(route, limiter, 'gateway'), methods=['POST'], include_in_schema=False)
            self.dynamic_routes[route.name] = route.to_route()

    def get_dynamic_route(self, route_name: str) -> Optional[Route]:
        return self.dynamic_routes.get(route_name)