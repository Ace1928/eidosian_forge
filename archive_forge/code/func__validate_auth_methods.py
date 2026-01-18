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
def _validate_auth_methods(self):
    for method_name in self.get_method_names():
        if method_name not in self.auth['identity']:
            raise exception.ValidationError(attribute=method_name, target='identity')
    for method_name in self.get_method_names():
        if method_name not in AUTH_METHODS:
            raise exception.AuthMethodNotSupported()