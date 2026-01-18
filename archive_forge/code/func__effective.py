import flask
from keystone.common import provider_api
from keystone.common import rbac_enforcer
from keystone import exception
from keystone.i18n import _
from keystone.server import flask as ks_flask
@property
def _effective(self):
    return self.query_filter_is_true('effective')