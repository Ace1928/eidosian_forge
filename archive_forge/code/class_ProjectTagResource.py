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
class ProjectTagResource(_ProjectTagResourceBase):

    def get(self, project_id, value):
        """Get information for a single tag associated with a given project.

        GET /v3/projects/{project_id}/tags/{value}
        """
        ENFORCER.enforce_call(action='identity:get_project_tag', build_target=_build_project_target_enforcement)
        PROVIDERS.resource_api.get_project_tag(project_id, value)
        return (None, http.client.NO_CONTENT)

    def put(self, project_id, value):
        """Add a single tag to a project.

        PUT /v3/projects/{project_id}/tags/{value}
        """
        ENFORCER.enforce_call(action='identity:create_project_tag', build_target=_build_project_target_enforcement)
        validation.lazy_validate(schema.project_tag_create, value)
        tags = PROVIDERS.resource_api.list_project_tags(project_id)
        tags.append(value)
        validation.lazy_validate(schema.project_tags_update, tags)
        PROVIDERS.resource_api.create_project_tag(project_id, value, initiator=self.audit_initiator)
        url = '/'.join((ks_flask.base_url(), project_id, 'tags', value))
        response = flask.make_response('', http.client.CREATED)
        response.headers['Location'] = url
        return response

    def delete(self, project_id, value):
        """Delete a single tag from a project.

        /v3/projects/{project_id}/tags/{value}
        """
        ENFORCER.enforce_call(action='identity:delete_project_tag', build_target=_build_project_target_enforcement)
        PROVIDERS.resource_api.delete_project_tag(project_id, value)
        return (None, http.client.NO_CONTENT)