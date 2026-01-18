import collections
import inspect
import logging
import warnings
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Union
from fastapi import APIRouter, FastAPI
import ray
from ray import cloudpickle
from ray._private.serialization import pickle_dumps
from ray.dag import DAGNode
from ray.serve._private.config import DeploymentConfig, ReplicaConfig
from ray.serve._private.constants import (
from ray.serve._private.deployment_graph_build import build as pipeline_build
from ray.serve._private.deployment_graph_build import (
from ray.serve._private.http_util import (
from ray.serve._private.usage import ServeUsageTag
from ray.serve._private.utils import (
from ray.serve.config import (
from ray.serve.context import (
from ray.serve.deployment import Application, Deployment
from ray.serve.exceptions import RayServeException
from ray.serve.handle import DeploymentHandle
from ray.serve.multiplex import _ModelMultiplexWrapper
from ray.serve.schema import LoggingConfig, ServeInstanceDetails, ServeStatus
from ray.util.annotations import Deprecated, DeveloperAPI, PublicAPI
from ray.serve._private import api as _private_api  # isort:skip
def _multiplex_decorator(func: Callable):

    @wraps(func)
    async def _multiplex_wrapper(*args):
        args_check_error_msg = 'Functions decorated with `@serve.multiplexed` must take exactly onethe multiplexed model ID (str), but got {}'
        if not args:
            raise TypeError(args_check_error_msg.format('no arguments are provided.'))
        self = extract_self_if_method_call(args, func)
        if self is None:
            if len(args) != 1:
                raise TypeError(args_check_error_msg.format('more than one arguments.'))
            multiplex_object = func
            model_id = args[0]
        else:
            if len(args) != 2:
                raise TypeError(args_check_error_msg.format('more than one arguments.'))
            multiplex_object = self
            model_id = args[1]
        multiplex_attr = '__serve_multiplex_wrapper'
        if not hasattr(multiplex_object, multiplex_attr):
            model_multiplex_wrapper = _ModelMultiplexWrapper(func, self, max_num_models_per_replica)
            setattr(multiplex_object, multiplex_attr, model_multiplex_wrapper)
        else:
            model_multiplex_wrapper = getattr(multiplex_object, multiplex_attr)
        return await model_multiplex_wrapper.load_model(model_id)
    return _multiplex_wrapper