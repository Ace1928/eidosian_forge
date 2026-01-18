import flask
import flask_restful
import functools
import http.client
from keystone.common import json_home
from keystone.common import provider_api
from keystone.common import rbac_enforcer
from keystone import exception
from keystone.server import flask as ks_flask
class SystemUsersResource(flask_restful.Resource):

    def get(self, user_id, role_id):
        """Check if a user has a specific role on the system.

        GET/HEAD /system/users/{user_id}/roles/{role_id}
        """
        ENFORCER.enforce_call(action='identity:check_system_grant_for_user', build_target=_build_enforcement_target)
        PROVIDERS.assignment_api.check_system_grant_for_user(user_id, role_id)
        return (None, http.client.NO_CONTENT)

    def put(self, user_id, role_id):
        """Grant a role to a user on the system.

        PUT /system/users/{user_id}/roles/{role_id}
        """
        ENFORCER.enforce_call(action='identity:create_system_grant_for_user', build_target=_build_enforcement_target)
        PROVIDERS.assignment_api.create_system_grant_for_user(user_id, role_id)
        return (None, http.client.NO_CONTENT)

    def delete(self, user_id, role_id):
        """Revoke a role from user on the system.

        DELETE /system/users/{user_id}/roles/{role_id}
        """
        ENFORCER.enforce_call(action='identity:revoke_system_grant_for_user', build_target=functools.partial(_build_enforcement_target, allow_non_existing=True))
        PROVIDERS.assignment_api.delete_system_grant_for_user(user_id, role_id)
        return (None, http.client.NO_CONTENT)