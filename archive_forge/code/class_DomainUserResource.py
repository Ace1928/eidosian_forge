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
class DomainUserResource(ks_flask.ResourceBase):
    member_key = 'grant'
    collection_key = 'grants'

    def get(self, domain_id=None, user_id=None, role_id=None):
        """Check if a user has a specific role on the domain.

        GET/HEAD /v3/domains/{domain_id}/users/{user_id}/roles/{role_id}
        """
        ENFORCER.enforce_call(action='identity:check_grant', build_target=_build_enforcement_target)
        PROVIDERS.assignment_api.get_grant(role_id, domain_id=domain_id, user_id=user_id, inherited_to_projects=False)
        return (None, http.client.NO_CONTENT)

    def put(self, domain_id=None, user_id=None, role_id=None):
        """Create a role to a user on a domain.

        PUT /v3/domains/{domain_id}/users/{user_id}/roles/{role_id}
        """
        ENFORCER.enforce_call(action='identity:create_grant', build_target=_build_enforcement_target)
        PROVIDERS.assignment_api.create_grant(role_id, domain_id=domain_id, user_id=user_id, inherited_to_projects=False, initiator=self.audit_initiator)
        return (None, http.client.NO_CONTENT)

    def delete(self, domain_id=None, user_id=None, role_id=None):
        """Revoke a role from user on a domain.

        DELETE /v3/domains/{domain_id}/users/{user_id}/roles/{role_id}
        """
        ENFORCER.enforce_call(action='identity:revoke_grant', build_target=functools.partial(_build_enforcement_target, allow_non_existing=True))
        PROVIDERS.assignment_api.delete_grant(role_id, domain_id=domain_id, user_id=user_id, inherited_to_projects=False, initiator=self.audit_initiator)
        return (None, http.client.NO_CONTENT)