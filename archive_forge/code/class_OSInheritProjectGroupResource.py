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
class OSInheritProjectGroupResource(flask_restful.Resource):

    def get(self, project_id, group_id, role_id):
        """Check for an inherited grant for a group on a project.

        GET/HEAD /OS-INHERIT/projects/{project_id}/groups/{group_id}
                 /roles/{role_id}/inherited_to_projects
        """
        ENFORCER.enforce_call(action='identity:check_grant', build_target=functools.partial(_build_enforcement_target_attr, project_id=project_id, group_id=group_id, role_id=role_id))
        PROVIDERS.assignment_api.get_grant(project_id=project_id, group_id=group_id, role_id=role_id, inherited_to_projects=True)
        return (None, http.client.NO_CONTENT)

    def put(self, project_id, group_id, role_id):
        """Create an inherited grant for a group on a project.

        PUT /OS-INHERIT/projects/{project_id}/groups/{group_id}
            /roles/{role_id}/inherited_to_projects
        """
        ENFORCER.enforce_call(action='identity:create_grant', build_target=functools.partial(_build_enforcement_target_attr, project_id=project_id, group_id=group_id, role_id=role_id))
        PROVIDERS.assignment_api.create_grant(project_id=project_id, group_id=group_id, role_id=role_id, inherited_to_projects=True)
        return (None, http.client.NO_CONTENT)

    def delete(self, project_id, group_id, role_id):
        """Revoke an inherited grant for a group on a project.

        DELETE /OS-INHERIT/projects/{project_id}/groups/{group_id}
               /roles/{role_id}/inherited_to_projects
        """
        ENFORCER.enforce_call(action='identity:revoke_grant', build_target=functools.partial(_build_enforcement_target_attr, project_id=project_id, group_id=group_id, role_id=role_id))
        PROVIDERS.assignment_api.delete_grant(project_id=project_id, group_id=group_id, role_id=role_id, inherited_to_projects=True)
        return (None, http.client.NO_CONTENT)