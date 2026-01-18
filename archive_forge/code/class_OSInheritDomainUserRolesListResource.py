import flask_restful
import functools
import http.client
from oslo_log import log
from keystone.api._shared import json_home_relations
from keystone.common import json_home
from keystone.common import provider_api
from keystone.common import rbac_enforcer
from keystone import exception
from keystone.server import flask as ks_flask
class OSInheritDomainUserRolesListResource(flask_restful.Resource):

    def get(self, domain_id, user_id):
        """List roles (inherited) for a user on a domain.

        GET/HEAD /OS-INHERIT/domains/{domain_id}/users/{user_id}
                 /roles/inherited_to_projects
        """
        ENFORCER.enforce_call(action='identity:list_grants', build_target=functools.partial(_build_enforcement_target_attr, domain_id=domain_id, user_id=user_id))
        refs = PROVIDERS.assignment_api.list_grants(domain_id=domain_id, user_id=user_id, inherited_to_projects=True)
        return ks_flask.ResourceBase.wrap_collection(refs, collection_name='roles')