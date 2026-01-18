import flask
import flask_restful
import http.client
from keystone.common import json_home
from keystone.common import provider_api
from keystone.common import rbac_enforcer
from keystone.common import validation
from keystone import exception
from keystone.limit import schema
from keystone.server import flask as ks_flask
class LimitModelResource(flask_restful.Resource):

    def get(self):
        ENFORCER.enforce_call(action='identity:get_limit_model')
        model = PROVIDERS.unified_limit_api.get_model()
        return {'model': model}