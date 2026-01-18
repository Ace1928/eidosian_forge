import logging
import warnings
from enum import Enum
from typing import Any, Callable, List, Optional, Union
from ray._private.pydantic_compat import (
from ray._private.utils import import_attr
from ray.serve._private.constants import (
from ray.util.annotations import Deprecated, PublicAPI
@classmethod
def _to_deployment_mode(cls, v: Union['ProxyLocation', str]) -> DeploymentMode:
    if not isinstance(v, (cls, str)):
        raise TypeError(f'Must be a `ProxyLocation` or str, got: {type(v)}.')
    elif v == ProxyLocation.Disabled:
        return DeploymentMode.NoServer
    elif v == ProxyLocation.HeadOnly:
        return DeploymentMode.HeadOnly
    elif v == ProxyLocation.EveryNode:
        return DeploymentMode.EveryNode
    else:
        raise ValueError(f'Unrecognized `ProxyLocation`: {v}.')