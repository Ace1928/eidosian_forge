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
def _delete_domain_contents(self, domain_id):
    """Delete the contents of a domain.

        Before we delete a domain, we need to remove all the entities
        that are owned by it, i.e. Projects. To do this we
        call the delete function for these entities, which are
        themselves responsible for deleting any credentials and role grants
        associated with them as well as revoking any relevant tokens.

        """

    def _delete_projects(project, projects, examined):
        if project['id'] in examined:
            msg = 'Circular reference or a repeated entry found projects hierarchy - %(project_id)s.'
            LOG.error(msg, {'project_id': project['id']})
            return
        examined.add(project['id'])
        children = [proj for proj in projects if proj.get('parent_id') == project['id']]
        for proj in children:
            _delete_projects(proj, projects, examined)
        try:
            self._delete_project(project, initiator=None)
        except exception.ProjectNotFound:
            LOG.debug('Project %(projectid)s not found when deleting domain contents for %(domainid)s, continuing with cleanup.', {'projectid': project['id'], 'domainid': domain_id})
    proj_refs = self.list_projects_in_domain(domain_id)
    roots = [x for x in proj_refs if x.get('parent_id') == domain_id]
    examined = set()
    for project in roots:
        _delete_projects(project, proj_refs, examined)