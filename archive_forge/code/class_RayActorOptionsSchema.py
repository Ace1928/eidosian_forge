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
class RayActorOptionsSchema(BaseModel):
    """Options with which to start a replica actor."""
    runtime_env: dict = Field(default={}, description="This deployment's runtime_env. working_dir and py_modules may contain only remote URIs.")
    num_cpus: float = Field(default=None, description="The number of CPUs required by the deployment's application per replica. This is the same as a ray actor's num_cpus. Uses a default if null.", ge=0)
    num_gpus: float = Field(default=None, description="The number of GPUs required by the deployment's application per replica. This is the same as a ray actor's num_gpus. Uses a default if null.", ge=0)
    memory: float = Field(default=None, description='Restrict the heap memory usage of each replica. Uses a default if null.', ge=0)
    object_store_memory: float = Field(default=None, description='Restrict the object store memory used per replica when creating objects. Uses a default if null.', ge=0)
    resources: Dict = Field(default={}, description='The custom resources required by each replica.')
    accelerator_type: str = Field(default=None, description='Forces replicas to run on nodes with the specified accelerator type.See :ref:`accelerator types <accelerator_types>`.')

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