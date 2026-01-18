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
class _ProjectGrantResourceBase(ks_flask.ResourceBase):
    collection_key = 'roles'
    member_key = 'role'
    get_member_from_driver = PROVIDERS.deferred_provider_lookup(api='role_api', method='get_role')

    @staticmethod
    def _check_if_inherited():
        return flask.request.path.endswith('/inherited_to_projects')

    @staticmethod
    def _build_enforcement_target_attr(role_id=None, user_id=None, group_id=None, domain_id=None, project_id=None, allow_non_existing=False):
        ref = {}
        if role_id:
            ref['role'] = PROVIDERS.role_api.get_role(role_id)
        try:
            if user_id:
                ref['user'] = PROVIDERS.identity_api.get_user(user_id)
            else:
                ref['group'] = PROVIDERS.identity_api.get_group(group_id)
        except (exception.UserNotFound, exception.GroupNotFound):
            if not allow_non_existing:
                raise
        if domain_id:
            ref['domain'] = PROVIDERS.resource_api.get_domain(domain_id)
        elif project_id:
            ref['project'] = PROVIDERS.resource_api.get_project(project_id)
        return ref