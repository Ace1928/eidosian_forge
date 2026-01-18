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
def _resolve_param_as_builtin(self, builtin_name, builtins):
    if builtin_name not in EndpointResolverBuiltins.__members__.values():
        raise UnknownEndpointResolutionBuiltInName(name=builtin_name)
    return builtins.get(builtin_name)