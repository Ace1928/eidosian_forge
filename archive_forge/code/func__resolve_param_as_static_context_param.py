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
def _resolve_param_as_static_context_param(self, param_name, operation_model):
    static_ctx_params = self._get_static_context_params(operation_model)
    return static_ctx_params.get(param_name)