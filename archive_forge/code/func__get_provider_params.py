import copy
import logging
import re
from enum import Enum
from botocore import UNSIGNED, xform_name
from botocore.auth import AUTH_TYPE_MAPS, HAS_CRT
from botocore.crt import CRT_SUPPORTED_AUTH_TYPES
from botocore.endpoint_provider import EndpointProvider
from botocore.exceptions import (
from botocore.utils import ensure_boolean, instance_cache
def _get_provider_params(self, operation_model, call_args, request_context):
    """Resolve a value for each parameter defined in the service's ruleset

        The resolution order for parameter values is:
        1. Operation-specific static context values from the service definition
        2. Operation-specific dynamic context values from API parameters
        3. Client-specific context parameters
        4. Built-in values such as region, FIPS usage, ...
        """
    provider_params = {}
    customized_builtins = self._get_customized_builtins(operation_model, call_args, request_context)
    for param_name, param_def in self._param_definitions.items():
        param_val = self._resolve_param_from_context(param_name=param_name, operation_model=operation_model, call_args=call_args)
        if param_val is None and param_def.builtin is not None:
            param_val = self._resolve_param_as_builtin(builtin_name=param_def.builtin, builtins=customized_builtins)
        if param_val is not None:
            provider_params[param_name] = param_val
    return provider_params