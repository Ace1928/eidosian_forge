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
def _retrieve_variant_data(self, endpoint_data, tags):
    variants = endpoint_data.get('variants', [])
    for variant in variants:
        if set(variant['tags']) == set(tags):
            result = variant.copy()
            return result