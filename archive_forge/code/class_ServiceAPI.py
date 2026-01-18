import http.client
from keystone.catalog import schema
from keystone.common import provider_api
from keystone.common import rbac_enforcer
from keystone.common import validation
from keystone.server import flask as ks_flask
class ServiceAPI(ks_flask.APIBase):
    _name = 'services'
    _import_name = __name__
    resources = [ServicesResource]
    resource_mapping = []