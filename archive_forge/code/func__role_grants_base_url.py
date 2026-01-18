from debtcollector import removals
from keystoneclient import base
from keystoneclient import exceptions
from keystoneclient.i18n import _
def _role_grants_base_url(self, user, group, system, domain, project, use_inherit_extension):
    params = {}
    if project:
        params['project_id'] = base.getid(project)
        base_url = '/projects/%(project_id)s'
    elif domain:
        params['domain_id'] = base.getid(domain)
        base_url = '/domains/%(domain_id)s'
    elif system:
        if system == 'all':
            base_url = '/system'
        else:
            msg = _("Only a system scope of 'all' is currently supported")
            raise exceptions.ValidationError(msg)
    if use_inherit_extension:
        base_url = '/OS-INHERIT' + base_url
    if user:
        params['user_id'] = base.getid(user)
        base_url += '/users/%(user_id)s'
    elif group:
        params['group_id'] = base.getid(group)
        base_url += '/groups/%(group_id)s'
    return base_url % params