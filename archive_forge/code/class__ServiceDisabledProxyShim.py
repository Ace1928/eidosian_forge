import typing as ty
import warnings
import os_service_types
from openstack import _log
from openstack import exceptions
from openstack import proxy as proxy_mod
from openstack import warnings as os_warnings
class _ServiceDisabledProxyShim:

    def __init__(self, service_type, reason):
        self.service_type = service_type
        self.reason = reason

    def __getattr__(self, item):
        raise exceptions.ServiceDisabledException("Service '{service_type}' is disabled because its configuration could not be loaded. {reason}".format(service_type=self.service_type, reason=self.reason or ''))