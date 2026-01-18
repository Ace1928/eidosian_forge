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
def _build_enforcement_target_attr(role_id=None, user_id=None, group_id=None, project_id=None, domain_id=None, allow_non_existing=False):
    """Check protection for role grant APIs.

    The policy rule might want to inspect attributes of any of the entities
    involved in the grant.  So we get these and pass them to the
    check_protection() handler in the controller.

    """
    target = {}
    if role_id:
        try:
            target['role'] = PROVIDERS.role_api.get_role(role_id)
        except exception.RoleNotFound:
            LOG.info('Role (%(role_id)s) not found, Enforcement target of `role` remaind empty', {'role_id': role_id})
            target['role'] = {}
    if user_id:
        try:
            target['user'] = PROVIDERS.identity_api.get_user(user_id)
        except exception.UserNotFound:
            if not allow_non_existing:
                LOG.info('User (%(user_id)s) was not found. Enforcement target of `user` remains empty.', {'user_id': user_id})
                target['user'] = {}
    else:
        try:
            target['group'] = PROVIDERS.identity_api.get_group(group_id)
        except exception.GroupNotFound:
            if not allow_non_existing:
                LOG.info('Group (%(group_id)s) was not found. Enforcement target of `group` remains empty.', {'group_id': group_id})
                target['group'] = {}
    if domain_id:
        try:
            target['domain'] = PROVIDERS.resource_api.get_domain(domain_id)
        except exception.DomainNotFound:
            LOG.info('Domain (%(domain_id)s) was not found. Enforcement target of `domain` remains empty.', {'domain_id': domain_id})
            target['domain'] = {}
    elif project_id:
        try:
            target['project'] = PROVIDERS.resource_api.get_project(project_id)
        except exception.ProjectNotFound:
            LOG.info('Project (%(project_id)s) was not found. Enforcement target of `project` remains empty.', {'project_id': project_id})
            target['project'] = {}
    return target