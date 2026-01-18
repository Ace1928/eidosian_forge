import functools
import flask
import http.client
from keystone.common import json_home
from keystone.common import provider_api
from keystone.common import rbac_enforcer
from keystone.common import validation
import keystone.conf
from keystone import exception
from keystone.i18n import _
from keystone.resource import schema
from keystone.server import flask as ks_flask
class ProjectTagsResource(_ProjectTagResourceBase):

    def get(self, project_id):
        """List tags associated with a given project.

        GET /v3/projects/{project_id}/tags
        """
        ENFORCER.enforce_call(action='identity:list_project_tags', build_target=_build_project_target_enforcement)
        ref = PROVIDERS.resource_api.list_project_tags(project_id)
        return self.wrap_member(ref)

    def put(self, project_id):
        """Update all tags associated with a given project.

        PUT /v3/projects/{project_id}/tags
        """
        ENFORCER.enforce_call(action='identity:update_project_tags', build_target=_build_project_target_enforcement)
        tags = self.request_body_json.get('tags', {})
        validation.lazy_validate(schema.project_tags_update, tags)
        ref = PROVIDERS.resource_api.update_project_tags(project_id, tags, initiator=self.audit_initiator)
        return self.wrap_member(ref)

    def delete(self, project_id):
        """Delete all tags associated with a given project.

        DELETE /v3/projects/{project_id}/tags
        """
        ENFORCER.enforce_call(action='identity:delete_project_tags', build_target=_build_project_target_enforcement)
        PROVIDERS.resource_api.update_project_tags(project_id, [])
        return (None, http.client.NO_CONTENT)