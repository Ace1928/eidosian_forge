import abc
import weakref
from keystoneauth1 import exceptions
from keystoneauth1.identity import generic
from keystoneauth1 import plugin
from oslo_config import cfg
from oslo_utils import excutils
import requests
from heat.common import config
from heat.common import exception as heat_exception
def _get_region_name(self):
    reg = self.context.region_name or cfg.CONF.region_name_for_services
    shared_services_region_name = cfg.CONF.region_name_for_shared_services
    shared_services_types = cfg.CONF.shared_services_types
    if shared_services_region_name:
        if set(self.service_types) & set(shared_services_types):
            reg = shared_services_region_name
    return reg