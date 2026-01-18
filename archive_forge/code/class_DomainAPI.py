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
class DomainAPI(ks_flask.APIBase):
    CONFIG_GROUP = json_home.build_v3_parameter_relation('config_group')
    CONFIG_OPTION = json_home.build_v3_parameter_relation('config_option')
    _name = 'domains'
    _import_name = __name__
    resources = [DomainResource]
    resource_mapping = [ks_flask.construct_resource_map(resource=DomainConfigResource, url='/domains/<string:domain_id>/config', resource_kwargs={}, rel='domain_config', path_vars={'domain_id': json_home.Parameters.DOMAIN_ID}), ks_flask.construct_resource_map(resource=DomainConfigGroupResource, url='/domains/<string:domain_id>/config/<string:group>', resource_kwargs={}, rel='domain_config_group', path_vars={'domain_id': json_home.Parameters.DOMAIN_ID, 'group': CONFIG_GROUP}), ks_flask.construct_resource_map(resource=DomainConfigOptionResource, url='/domains/<string:domain_id>/config/<string:group>/<string:option>', resource_kwargs={}, rel='domain_config_option', path_vars={'domain_id': json_home.Parameters.DOMAIN_ID, 'group': CONFIG_GROUP, 'option': CONFIG_OPTION}), ks_flask.construct_resource_map(resource=DefaultConfigResource, url='/domains/config/default', resource_kwargs={}, rel='domain_config_default', path_vars={}), ks_flask.construct_resource_map(resource=DefaultConfigGroupResource, url='/domains/config/<string:group>/default', resource_kwargs={}, rel='domain_config_default_group', path_vars={'group': CONFIG_GROUP}), ks_flask.construct_resource_map(resource=DefaultConfigOptionResource, url='/domains/config/<string:group>/<string:option>/default', resource_kwargs={}, rel='domain_config_default_option', path_vars={'group': CONFIG_GROUP, 'option': CONFIG_OPTION}), ks_flask.construct_resource_map(resource=DomainUserListResource, url='/domains/<string:domain_id>/users/<string:user_id>/roles', resource_kwargs={}, rel='domain_user_roles', path_vars={'domain_id': json_home.Parameters.DOMAIN_ID, 'user_id': json_home.Parameters.USER_ID}), ks_flask.construct_resource_map(resource=DomainUserResource, url='/domains/<string:domain_id>/users/<string:user_id>/roles/<string:role_id>', resource_kwargs={}, rel='domain_user_role', path_vars={'domain_id': json_home.Parameters.DOMAIN_ID, 'user_id': json_home.Parameters.USER_ID, 'role_id': json_home.Parameters.ROLE_ID}), ks_flask.construct_resource_map(resource=DomainGroupListResource, url='/domains/<string:domain_id>/groups/<string:group_id>/roles', resource_kwargs={}, rel='domain_group_roles', path_vars={'domain_id': json_home.Parameters.DOMAIN_ID, 'group_id': json_home.Parameters.GROUP_ID}), ks_flask.construct_resource_map(resource=DomainGroupResource, url='/domains/<string:domain_id>/groups/<string:group_id>/roles/<string:role_id>', resource_kwargs={}, rel='domain_group_role', path_vars={'domain_id': json_home.Parameters.DOMAIN_ID, 'group_id': json_home.Parameters.GROUP_ID, 'role_id': json_home.Parameters.ROLE_ID})]