import base64
import secrets
import uuid
import flask
import http.client
from oslo_serialization import jsonutils
from werkzeug import exceptions
from keystone.api._shared import json_home_relations
from keystone.application_credential import schema as app_cred_schema
from keystone.common import json_home
from keystone.common import provider_api
from keystone.common import rbac_enforcer
from keystone.common import utils
from keystone.common import validation
import keystone.conf
from keystone import exception as ks_exception
from keystone.i18n import _
from keystone.identity import schema
from keystone import notifications
from keystone.server import flask as ks_flask
class UserAPI(ks_flask.APIBase):
    _name = 'users'
    _import_name = __name__
    resources = [UserResource]
    resource_mapping = [ks_flask.construct_resource_map(resource=UserChangePasswordResource, url='/users/<string:user_id>/password', resource_kwargs={}, rel='user_change_password', path_vars={'user_id': json_home.Parameters.USER_ID}), ks_flask.construct_resource_map(resource=UserGroupsResource, url='/users/<string:user_id>/groups', resource_kwargs={}, rel='user_groups', path_vars={'user_id': json_home.Parameters.USER_ID}), ks_flask.construct_resource_map(resource=UserProjectsResource, url='/users/<string:user_id>/projects', resource_kwargs={}, rel='user_projects', path_vars={'user_id': json_home.Parameters.USER_ID}), ks_flask.construct_resource_map(resource=UserOSEC2CredentialsResourceListCreate, url='/users/<string:user_id>/credentials/OS-EC2', resource_kwargs={}, rel='user_credentials', resource_relation_func=json_home_relations.os_ec2_resource_rel_func, path_vars={'user_id': json_home.Parameters.USER_ID}), ks_flask.construct_resource_map(resource=UserOSEC2CredentialsResourceGetDelete, url='/users/<string:user_id>/credentials/OS-EC2/<string:credential_id>', resource_kwargs={}, rel='user_credential', resource_relation_func=json_home_relations.os_ec2_resource_rel_func, path_vars={'credential_id': json_home.build_v3_parameter_relation('credential_id'), 'user_id': json_home.Parameters.USER_ID}), ks_flask.construct_resource_map(resource=OAuth1ListAccessTokensResource, url='/users/<string:user_id>/OS-OAUTH1/access_tokens', resource_kwargs={}, rel='user_access_tokens', resource_relation_func=json_home_relations.os_oauth1_resource_rel_func, path_vars={'user_id': json_home.Parameters.USER_ID}), ks_flask.construct_resource_map(resource=OAuth1AccessTokenCRUDResource, url='/users/<string:user_id>/OS-OAUTH1/access_tokens/<string:access_token_id>', resource_kwargs={}, rel='user_access_token', resource_relation_func=json_home_relations.os_oauth1_resource_rel_func, path_vars={'access_token_id': ACCESS_TOKEN_ID_PARAMETER_RELATION, 'user_id': json_home.Parameters.USER_ID}), ks_flask.construct_resource_map(resource=OAuth1AccessTokenRoleListResource, url='/users/<string:user_id>/OS-OAUTH1/access_tokens/<string:access_token_id>/roles', resource_kwargs={}, rel='user_access_token_roles', resource_relation_func=json_home_relations.os_oauth1_resource_rel_func, path_vars={'access_token_id': ACCESS_TOKEN_ID_PARAMETER_RELATION, 'user_id': json_home.Parameters.USER_ID}), ks_flask.construct_resource_map(resource=OAuth1AccessTokenRoleResource, url='/users/<string:user_id>/OS-OAUTH1/access_tokens/<string:access_token_id>/roles/<string:role_id>', resource_kwargs={}, rel='user_access_token_role', resource_relation_func=json_home_relations.os_oauth1_resource_rel_func, path_vars={'access_token_id': ACCESS_TOKEN_ID_PARAMETER_RELATION, 'role_id': json_home.Parameters.ROLE_ID, 'user_id': json_home.Parameters.USER_ID}), ks_flask.construct_resource_map(resource=UserAppCredListCreateResource, url='/users/<string:user_id>/application_credentials', resource_kwargs={}, rel='application_credentials', path_vars={'user_id': json_home.Parameters.USER_ID}), ks_flask.construct_resource_map(resource=UserAppCredGetDeleteResource, url='/users/<string:user_id>/application_credentials/<string:application_credential_id>', resource_kwargs={}, rel='application_credential', path_vars={'user_id': json_home.Parameters.USER_ID, 'application_credential_id': json_home.Parameters.APPLICATION_CRED_ID}), ks_flask.construct_resource_map(resource=UserAccessRuleListResource, url='/users/<string:user_id>/access_rules', resource_kwargs={}, rel='access_rules', path_vars={'user_id': json_home.Parameters.USER_ID}), ks_flask.construct_resource_map(resource=UserAccessRuleGetDeleteResource, url='/users/<string:user_id>/access_rules/<string:access_rule_id>', resource_kwargs={}, rel='access_rule', path_vars={'user_id': json_home.Parameters.USER_ID, 'access_rule_id': json_home.Parameters.ACCESS_RULE_ID})]