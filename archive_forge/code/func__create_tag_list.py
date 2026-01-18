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
def _create_tag_list(self, use_dualstack_endpoint, use_fips_endpoint):
    tags = []
    if use_dualstack_endpoint:
        tags.append('dualstack')
    if use_fips_endpoint:
        tags.append('fips')
    return tags