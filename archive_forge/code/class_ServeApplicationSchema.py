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
class ServeApplicationSchema(BaseModel):
    """
    Describes one Serve application, and currently can also be used as a standalone
    config to deploy a single application to a Ray cluster.
    """
    name: str = Field(default=SERVE_DEFAULT_APP_NAME, description='Application name, the name should be unique within the serve instance')
    route_prefix: Optional[str] = Field(default='/', description="Route prefix for HTTP requests. If not provided, it will useroute_prefix of the ingress deployment. By default, the ingress route prefix is '/'.")
    import_path: str = Field(..., description='An import path to a bound deployment node. Should be of the form "module.submodule_1...submodule_n.dag_node". This is equivalent to "from module.submodule_1...submodule_n import dag_node". Only works with Python applications. This field is REQUIRED when deploying Serve config to a Ray cluster.')
    runtime_env: dict = Field(default={}, description='The runtime_env that the deployment graph will be run in. Per-deployment runtime_envs will inherit from this. working_dir and py_modules may contain only remote URIs.')
    host: str = Field(default='0.0.0.0', description='Host for HTTP servers to listen on. Defaults to "0.0.0.0", which exposes Serve publicly. Cannot be updated once your Serve application has started running. The Serve application must be shut down and restarted with the new host instead.')
    port: int = Field(default=8000, description='Port for HTTP server. Defaults to 8000. Cannot be updated once your Serve application has started running. The Serve application must be shut down and restarted with the new port instead.')
    deployments: List[DeploymentSchema] = Field(default=[], description='Deployment options that override options specified in the code.')
    args: Dict = Field(default={}, description='Arguments that will be passed to the application builder.')
    logging_config: LoggingConfig = Field(default=None, description='Logging config for configuring serve application logs.')

    @property
    def deployment_names(self) -> List[str]:
        return [d.name for d in self.deployments]

    @validator('runtime_env')
    def runtime_env_contains_remote_uris(cls, v):
        if v is None:
            return
        uris = v.get('py_modules', [])
        if 'working_dir' in v and v['working_dir'] not in uris:
            uris.append(v['working_dir'])
        for uri in uris:
            if uri is not None:
                try:
                    parse_uri(uri)
                except ValueError as e:
                    raise ValueError(f'runtime_envs in the Serve config support only remote URIs in working_dir and py_modules. Got error when parsing URI: {e}')
        return v

    @validator('import_path')
    def import_path_format_valid(cls, v: str):
        if v is None:
            return
        if ':' in v:
            if v.count(':') > 1:
                raise ValueError(f'Got invalid import path "{v}". An import path may have at most one colon.')
            if v.rfind(':') == 0 or v.rfind(':') == len(v) - 1:
                raise ValueError(f'Got invalid import path "{v}". An import path may not start or end with a colon.')
            return v
        else:
            if v.count('.') < 1:
                raise ValueError(f'Got invalid import path "{v}". An import path must contain at least on dot or colon separating the module (and potentially submodules) from the deployment graph. E.g.: "module.deployment_graph".')
            if v.rfind('.') == 0 or v.rfind('.') == len(v) - 1:
                raise ValueError(f'Got invalid import path "{v}". An import path may not start or end with a dot.')
        return v

    @staticmethod
    def get_empty_schema_dict() -> Dict:
        """Returns an empty app schema dictionary.

        Schema can be used as a representation of an empty Serve application config.
        """
        return {'import_path': '', 'runtime_env': {}, 'deployments': []}