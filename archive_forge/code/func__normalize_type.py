import copy
import os_service_types.data
from os_service_types import exc
def _normalize_type(service_type):
    if service_type:
        return service_type.replace('_', '-')