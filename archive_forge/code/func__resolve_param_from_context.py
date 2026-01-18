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
def _resolve_param_from_context(self, param_name, operation_model, call_args):
    static = self._resolve_param_as_static_context_param(param_name, operation_model)
    if static is not None:
        return static
    dynamic = self._resolve_param_as_dynamic_context_param(param_name, operation_model, call_args)
    if dynamic is not None:
        return dynamic
    return self._resolve_param_as_client_context_param(param_name)