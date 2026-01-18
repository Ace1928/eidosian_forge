from functools import partial
from oslo_log import log
import stevedore
from keystone.common import driver_hints
from keystone.common import provider_api
from keystone.common import utils
import keystone.conf
from keystone import exception
from keystone.i18n import _
from keystone.identity.backends import resource_options as ro
def _lookup_project(self, project_info):
    project_id = project_info.get('id')
    project_name = project_info.get('name')
    try:
        if project_name:
            if CONF.resource.project_name_url_safe == 'strict' and utils.is_not_url_safe(project_name):
                msg = 'Project name cannot contain reserved characters.'
                tr_msg = _('Project name cannot contain reserved characters.')
                LOG.warning(msg)
                raise exception.Unauthorized(message=tr_msg)
            if 'domain' not in project_info:
                raise exception.ValidationError(attribute='domain', target='project')
            domain_ref = self._lookup_domain(project_info['domain'])
            project_ref = PROVIDERS.resource_api.get_project_by_name(project_name, domain_ref['id'])
        else:
            project_ref = PROVIDERS.resource_api.get_project(project_id)
            domain_id = project_ref['domain_id']
            if not domain_id:
                raise exception.ProjectNotFound(project_id=project_id)
            self._lookup_domain({'id': domain_id})
    except exception.ProjectNotFound as e:
        LOG.warning(e)
        raise exception.Unauthorized(e)
    self._assert_project_is_enabled(project_ref)
    return project_ref