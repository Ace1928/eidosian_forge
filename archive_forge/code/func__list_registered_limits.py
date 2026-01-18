import flask
import http.client
from keystone.common import json_home
from keystone.common import provider_api
from keystone.common import rbac_enforcer
from keystone.common import validation
from keystone.limit import schema
from keystone.server import flask as ks_flask
def _list_registered_limits(self):
    filters = ['service_id', 'region_id', 'resource_name']
    ENFORCER.enforce_call(action='identity:list_registered_limits', filters=filters)
    hints = self.build_driver_hints(filters)
    refs = PROVIDERS.unified_limit_api.list_registered_limits(hints)
    return self.wrap_collection(refs, hints=hints)