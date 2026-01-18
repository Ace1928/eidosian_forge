import flask
import http.client
from keystone.common import json_home
from keystone.common import provider_api
from keystone.common import rbac_enforcer
from keystone.common import validation
from keystone.limit import schema
from keystone.server import flask as ks_flask
class RegisteredLimitsAPI(ks_flask.APIBase):
    _name = 'registered_limit'
    _import_name = __name__
    resources = [RegisteredLimitResource]
    resource_mapping = []