import inspect
import logging
from types import FunctionType
from typing import Any, Dict, Tuple, Union
import ray
from ray._private.pydantic_compat import is_subclass_of_base_model
from ray._private.resource_spec import HEAD_NODE_RESOURCE_NAME
from ray._private.usage import usage_lib
from ray.actor import ActorHandle
from ray.serve._private.client import ServeControllerClient
from ray.serve._private.constants import (
from ray.serve._private.controller import ServeController
from ray.serve.config import HTTPOptions, gRPCOptions
from ray.serve.context import _get_global_client, _set_global_client
from ray.serve.deployment import Application, Deployment
from ray.serve.exceptions import RayServeException
from ray.serve.schema import LoggingConfig
def _check_http_options(client: ServeControllerClient, http_options: Union[dict, HTTPOptions]) -> None:
    if http_options:
        client_http_options = client.http_config
        new_http_options = http_options if isinstance(http_options, HTTPOptions) else HTTPOptions.parse_obj(http_options)
        different_fields = []
        all_http_option_fields = new_http_options.__dict__
        for field in all_http_option_fields:
            if getattr(new_http_options, field) != getattr(client_http_options, field):
                different_fields.append(field)
        if len(different_fields):
            logger.warning(f'The new client HTTP config differs from the existing one in the following fields: {different_fields}. The new HTTP config is ignored.')