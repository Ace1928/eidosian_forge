import flask
import flask_restful
import http.client
from oslo_serialization import jsonutils
from oslo_log import log
from keystone.api._shared import authentication
from keystone.api._shared import json_home_relations
from keystone.common import provider_api
from keystone.common import rbac_enforcer
from keystone.common import render_token
from keystone.common import validation
import keystone.conf
from keystone import exception
from keystone.federation import schema
from keystone.federation import utils
from keystone.server import flask as ks_flask
def _list_mappings(self):
    """List mappings.

        HEAD/GET /OS-FEDERATION/mappings
        """
    ENFORCER.enforce_call(action='identity:list_mappings')
    return self.wrap_collection(PROVIDERS.federation_api.list_mappings())