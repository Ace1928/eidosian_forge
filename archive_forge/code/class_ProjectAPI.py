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
class ProjectAPI(ks_flask.APIBase):
    _name = 'projects'
    _import_name = __name__
    resources = [ProjectResource]
    resource_mapping = [ks_flask.construct_resource_map(resource=ProjectTagsResource, url='/projects/<string:project_id>/tags', resource_kwargs={}, rel='project_tags', path_vars={'project_id': json_home.Parameters.PROJECT_ID}), ks_flask.construct_resource_map(resource=ProjectTagResource, url='/projects/<string:project_id>/tags/<string:value>', resource_kwargs={}, rel='project_tags', path_vars={'project_id': json_home.Parameters.PROJECT_ID, 'value': json_home.Parameters.TAG_VALUE}), ks_flask.construct_resource_map(resource=ProjectUserGrantResource, url='/projects/<string:project_id>/users/<string:user_id>/roles/<string:role_id>', resource_kwargs={}, rel='project_user_role', path_vars={'project_id': json_home.Parameters.PROJECT_ID, 'user_id': json_home.Parameters.USER_ID, 'role_id': json_home.Parameters.ROLE_ID}), ks_flask.construct_resource_map(resource=ProjectUserListGrantResource, url='/projects/<string:project_id>/users/<string:user_id>/roles', resource_kwargs={}, rel='project_user_roles', path_vars={'project_id': json_home.Parameters.PROJECT_ID, 'user_id': json_home.Parameters.USER_ID}), ks_flask.construct_resource_map(resource=ProjectGroupGrantResource, url='/projects/<string:project_id>/groups/<string:group_id>/roles/<string:role_id>', resource_kwargs={}, rel='project_group_role', path_vars={'project_id': json_home.Parameters.PROJECT_ID, 'group_id': json_home.Parameters.GROUP_ID, 'role_id': json_home.Parameters.ROLE_ID}), ks_flask.construct_resource_map(resource=ProjectGroupListGrantResource, url='/projects/<string:project_id>/groups/<string:group_id>/roles', resource_kwargs={}, rel='project_group_roles', path_vars={'project_id': json_home.Parameters.PROJECT_ID, 'group_id': json_home.Parameters.GROUP_ID})]