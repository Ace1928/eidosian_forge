import copy
import logging
from openstack.config import loader as config  # noqa
from openstack import connection
from oslo_utils import strutils
from osc_lib.api import auth
from osc_lib import exceptions
def _override_for(self, service_type):
    key = '%s_endpoint_override' % service_type.replace('-', '_')
    return self._cli_options.config.get(key)