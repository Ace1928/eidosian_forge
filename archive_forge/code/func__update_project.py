from oslo_log import log
from keystone import assignment
from keystone.common import cache
from keystone.common import driver_hints
from keystone.common import manager
from keystone.common import provider_api
from keystone.common.resource_options import options as ro_opt
from keystone.common import utils
import keystone.conf
from keystone import exception
from keystone.i18n import _
from keystone import notifications
from keystone.resource.backends import base
from keystone.token import provider as token_provider
def _update_project(self, project_id, project, initiator=None, cascade=False):
    original_project = self.driver.get_project(project_id)
    project = project.copy()
    self._require_matching_domain_id(project, original_project)
    if original_project['is_domain']:
        ro_opt.check_immutable_update(original_resource_ref=original_project, new_resource_ref=project, type='domain', resource_id=project_id)
        domain = self._get_domain_from_project(original_project)
        self.assert_domain_not_federated(project_id, domain)
        url_safe_option = CONF.resource.domain_name_url_safe
        exception_entity = 'Domain'
    else:
        ro_opt.check_immutable_update(original_resource_ref=original_project, new_resource_ref=project, type='project', resource_id=project_id)
        url_safe_option = CONF.resource.project_name_url_safe
        exception_entity = 'Project'
    project_name_changed = 'name' in project and project['name'] != original_project['name']
    if url_safe_option != 'off' and project_name_changed and utils.is_not_url_safe(project['name']):
        self._raise_reserved_character_exception(exception_entity, project['name'])
    elif project_name_changed:
        project['name'] = project['name'].strip()
    parent_id = original_project.get('parent_id')
    if 'parent_id' in project and project.get('parent_id') != parent_id:
        raise exception.ForbiddenNotSecurity(_('Update of `parent_id` is not allowed.'))
    if 'is_domain' in project and project['is_domain'] != original_project['is_domain']:
        raise exception.ValidationError(message=_('Update of `is_domain` is not allowed.'))
    original_project_enabled = original_project.get('enabled', True)
    project_enabled = project.get('enabled', True)
    if not original_project_enabled and project_enabled:
        self._assert_all_parents_are_enabled(project_id)
    if original_project_enabled and (not project_enabled):
        if not original_project.get('is_domain') and (not cascade) and (not self._check_whole_subtree_is_disabled(project_id)):
            raise exception.ForbiddenNotSecurity(_('Cannot disable project %(project_id)s since its subtree contains enabled projects.') % {'project_id': project_id})
        notifications.Audit.disabled(self._PROJECT, project_id, public=False)
        assignment.COMPUTED_ASSIGNMENTS_REGION.invalidate()
    if cascade:
        self._only_allow_enabled_to_update_cascade(project, original_project)
        self._update_project_enabled_cascade(project_id, project_enabled)
    try:
        project['is_domain'] = project.get('is_domain') or original_project['is_domain']
        ret = self.driver.update_project(project_id, project)
    except exception.Conflict:
        raise exception.Conflict(type='project', details=self._generate_project_name_conflict_msg(project))
    try:
        self.get_project.invalidate(self, project_id)
        self.get_project_by_name.invalidate(self, original_project['name'], original_project['domain_id'])
        if 'domain_id' in project and project['domain_id'] != original_project['domain_id']:
            assignment.COMPUTED_ASSIGNMENTS_REGION.invalidate()
    finally:
        notifications.Audit.updated(self._PROJECT, project_id, initiator)
        if original_project['is_domain']:
            notifications.Audit.updated(self._DOMAIN, project_id, initiator)
            if original_project_enabled and (not project_enabled):
                token_provider.TOKENS_REGION.invalidate()
                notifications.Audit.disabled(self._DOMAIN, project_id, public=False)
    return ret