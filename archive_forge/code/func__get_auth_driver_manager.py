from functools import partial
from oslo_log import log
import stevedore
from keystone.common import driver_hints
from keystone.common import provider_api
from keystone.common import utils
import keystone.conf
from keystone import exception
from keystone.i18n import _
from keystone.identity.backends import resource_options as ro
def _get_auth_driver_manager(namespace, plugin_name):
    return stevedore.DriverManager(namespace, plugin_name, invoke_on_load=True)