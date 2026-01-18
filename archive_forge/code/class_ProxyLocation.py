import logging
import warnings
from enum import Enum
from typing import Any, Callable, List, Optional, Union
from ray._private.pydantic_compat import (
from ray._private.utils import import_attr
from ray.serve._private.constants import (
from ray.util.annotations import Deprecated, PublicAPI
@PublicAPI(stability='stable')
class ProxyLocation(str, Enum):
    """Config for where to run proxies to receive ingress traffic to the cluster.

    Options:

        - Disabled: don't run proxies at all. This should be used if you are only
          making calls to your applications via deployment handles.
        - HeadOnly: only run a single proxy on the head node.
        - EveryNode: run a proxy on every node in the cluster that has at least one
          replica actor. This is the default.
    """
    Disabled = 'Disabled'
    HeadOnly = 'HeadOnly'
    EveryNode = 'EveryNode'

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