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
from keystone.identity import schema
from keystone import notifications
from keystone.server import flask as ks_flask
class GroupAPI(ks_flask.APIBase):
    _name = 'groups'
    _import_name = __name__
    resources = [GroupsResource]
    resource_mapping = [ks_flask.construct_resource_map(resource=GroupUsersResource, url='/groups/<string:group_id>/users', resource_kwargs={}, rel='group_users', path_vars={'group_id': json_home.Parameters.GROUP_ID}), ks_flask.construct_resource_map(resource=UserGroupCRUDResource, url='/groups/<string:group_id>/users/<string:user_id>', resource_kwargs={}, rel='group_user', path_vars={'group_id': json_home.Parameters.GROUP_ID, 'user_id': json_home.Parameters.USER_ID})]