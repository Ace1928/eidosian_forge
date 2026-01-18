import flask_restful
import http.client
from oslo_log import versionutils
from keystone.api._shared import json_home_relations
from keystone.common import json_home
from keystone.common import provider_api
from keystone.common import rbac_enforcer
from keystone.common import validation
from keystone.policy import schema
from keystone.server import flask as ks_flask
def _remove_legacy_ids(self, endpoints):
    for endpoint in endpoints:
        endpoint.pop('legacy_endpoint_id', None)