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
@instance_cache
def _get_static_context_params(self, operation_model):
    """Mapping of param names to static param value for an operation"""
    return {param.name: param.value for param in operation_model.static_context_parameters}