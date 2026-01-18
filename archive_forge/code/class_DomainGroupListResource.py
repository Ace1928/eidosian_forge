import flask
import flask_restful
import functools
import http.client
from keystone.common import json_home
from keystone.common import provider_api
from keystone.common import rbac_enforcer
from keystone.common import validation
import keystone.conf
from keystone import exception
from keystone.resource import schema
from keystone.server import flask as ks_flask
class DomainGroupListResource(flask_restful.Resource):

    def get(self, domain_id=None, group_id=None):
        """List all domain grats for a specific group.

        GET/HEAD /v3/domains/{domain_id}/groups/{group_id}/roles
        """
        ENFORCER.enforce_call(action='identity:list_grants', build_target=_build_enforcement_target)
        refs = PROVIDERS.assignment_api.list_grants(domain_id=domain_id, group_id=group_id, inherited_to_projects=False)
        return ks_flask.ResourceBase.wrap_collection(refs, collection_name='roles')