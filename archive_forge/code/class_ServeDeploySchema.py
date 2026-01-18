import logging
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union
from zlib import crc32
from ray._private.pydantic_compat import (
from ray._private.runtime_env.packaging import parse_uri
from ray.serve._private.common import (
from ray.serve._private.constants import (
from ray.serve._private.deployment_info import DeploymentInfo
from ray.serve._private.utils import DEFAULT
from ray.serve.config import ProxyLocation
from ray.util.annotations import PublicAPI
@PublicAPI(stability='stable')
class ServeDeploySchema(BaseModel):
    """
    Multi-application config for deploying a list of Serve applications to the Ray
    cluster.

    This is the request JSON schema for the v2 REST API
    `PUT "/api/serve/applications/"`.

    NOTE: This config allows extra parameters to make it forward-compatible (ie
          older versions of Serve are able to accept configs from a newer versions,
          simply ignoring new parameters)
    """
    proxy_location: ProxyLocation = Field(default=ProxyLocation.EveryNode, description='Config for where to run proxies for ingress traffic to the cluster.')
    http_options: HTTPOptionsSchema = Field(default=HTTPOptionsSchema(), description='Options to start the HTTP Proxy with.')
    grpc_options: gRPCOptionsSchema = Field(default=gRPCOptionsSchema(), description='Options to start the gRPC Proxy with.')
    logging_config: LoggingConfig = Field(default=None, description='Logging config for configuring serve components logs.')
    applications: List[ServeApplicationSchema] = Field(..., description='The set of applications to run on the Ray cluster.')
    target_capacity: Optional[float] = TARGET_CAPACITY_FIELD

    @validator('applications')
    def application_names_unique(cls, v):
        names = [app.name for app in v]
        duplicates = {f'"{name}"' for name in names if names.count(name) > 1}
        if len(duplicates):
            apps_str = ('application ' if len(duplicates) == 1 else 'applications ') + ', '.join(duplicates)
            raise ValueError(f'Found multiple configs for {apps_str}. Please remove all duplicates.')
        return v

    @validator('applications')
    def application_routes_unique(cls, v):
        routes = [app.route_prefix for app in v if app.route_prefix is not None]
        duplicates = {f'"{route}"' for route in routes if routes.count(route) > 1}
        if len(duplicates):
            routes_str = ('route prefix ' if len(duplicates) == 1 else 'route prefixes ') + ', '.join(duplicates)
            raise ValueError(f"Found duplicate applications for {routes_str}. Please ensure each application's route_prefix is unique.")
        return v

    @validator('applications')
    def application_names_nonempty(cls, v):
        for app in v:
            if len(app.name) == 0:
                raise ValueError('Application names must be nonempty.')
        return v

    @root_validator
    def nested_host_and_port(cls, values):
        for app_config in values.get('applications'):
            if 'host' in app_config.dict(exclude_unset=True):
                raise ValueError(f'Host "{app_config.host}" is set in the config for application `{app_config.name}`. Please remove it and set host in the top level deploy config only.')
            if 'port' in app_config.dict(exclude_unset=True):
                raise ValueError(f'Port {app_config.port} is set in the config for application `{app_config.name}`. Please remove it and set port in the top level deploy config only.')
        return values

    @staticmethod
    def get_empty_schema_dict() -> Dict:
        """Returns an empty deploy schema dictionary.

        Schema can be used as a representation of an empty Serve deploy config.
        """
        return {'applications': []}