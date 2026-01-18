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
def _get_service_config(self, key, service_type):
    config_dict = self.config.get(key)
    if not config_dict:
        return None
    if not isinstance(config_dict, dict):
        return config_dict
    for st in self._service_type_manager.get_all_types(service_type):
        if st in config_dict:
            return config_dict[st]