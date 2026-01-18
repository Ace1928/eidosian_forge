import copy
import os.path
import typing as ty
from urllib import parse
import warnings
from keystoneauth1 import discover
import keystoneauth1.exceptions.catalog
from keystoneauth1.loading import adapter as ks_load_adap
from keystoneauth1 import session as ks_session
import os_service_types
import requestsexceptions
from openstack import _log
from openstack.config import _util
from openstack.config import defaults as config_defaults
from openstack import exceptions
from openstack import proxy
from openstack import version as openstack_version
from openstack import warnings as os_warnings
def _get_endpoint_from_catalog(self, service_type, constructor):
    adapter = constructor(session=self.get_session(), service_type=self.get_service_type(service_type), service_name=self.get_service_name(service_type), interface=self.get_interface(service_type), region_name=self.get_region_name(service_type))
    return adapter.get_endpoint()