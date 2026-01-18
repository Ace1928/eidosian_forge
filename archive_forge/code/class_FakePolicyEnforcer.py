from cryptography import exceptions as crypto_exception
import glance_store as store
from unittest import mock
import urllib
from oslo_config import cfg
from oslo_policy import policy
from glance.async_.flows._internal_plugins import base_download
from glance.common import exception
from glance.common import store_utils
from glance.common import wsgi
import glance.context
import glance.db.simple.api as simple_db
class FakePolicyEnforcer(object):

    def __init__(self, *_args, **kwargs):
        self.rules = {}

    def enforce(self, _ctxt, action, target=None, **kwargs):
        """Raise Forbidden if a rule for given action is set to false."""
        if self.rules.get(action) is False:
            raise exception.Forbidden()

    def set_rules(self, rules):
        self.rules = rules