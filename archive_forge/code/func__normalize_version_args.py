import copy
import re
import urllib
import os_service_types
from keystoneauth1 import _utils as utils
from keystoneauth1 import exceptions
def _normalize_version_args(version, min_version, max_version, service_type=None):
    if service_type and _SERVICE_TYPES.is_known(service_type) and service_type[-1].isdigit() and (service_type[-2] == 'v'):
        implied_version = normalize_version_number(service_type[-1])
    else:
        implied_version = None
    if version and (min_version or max_version):
        raise ValueError('version is mutually exclusive with min_version and max_version')
    if version:
        min_version = normalize_version_number(version)
        max_version = (min_version[0], LATEST)
        if implied_version:
            if min_version[0] != implied_version[0]:
                raise exceptions.ImpliedVersionMismatch(service_type=service_type, implied=implied_version, given=version_to_string(version))
        return (min_version, max_version)
    if min_version == 'latest':
        if max_version not in (None, 'latest'):
            raise ValueError("min_version is 'latest' and max_version is {max_version} but is only allowed to be 'latest' or None".format(max_version=max_version))
        max_version = 'latest'
    min_version = min_version or None
    max_version = max_version or None
    if min_version:
        min_version = normalize_version_number(min_version)
        max_version = normalize_version_number(max_version or 'latest')
    if max_version:
        max_version = normalize_version_number(max_version)
    if None not in (min_version, max_version) and max_version < min_version:
        raise ValueError('min_version cannot be greater than max_version')
    if implied_version:
        if min_version:
            if min_version[0] != implied_version[0]:
                raise exceptions.ImpliedMinVersionMismatch(service_type=service_type, implied=implied_version, given=version_to_string(min_version))
        else:
            min_version = implied_version
        if max_version and max_version[0] != LATEST:
            if max_version[0] != implied_version[0]:
                raise exceptions.ImpliedMaxVersionMismatch(service_type=service_type, implied=implied_version, given=version_to_string(max_version))
        else:
            max_version = (implied_version[0], LATEST)
    return (min_version, max_version)