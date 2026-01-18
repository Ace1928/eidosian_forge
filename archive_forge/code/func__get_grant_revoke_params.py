from openstack.cloud import _utils
from openstack import exceptions
from openstack.identity.v3._proxy import Proxy
from openstack import utils
def _get_grant_revoke_params(self, role, user=None, group=None, project=None, domain=None, system=None):
    data = {}
    search_args = {}
    if domain:
        data['domain'] = self.identity.find_domain(domain, ignore_missing=False)
        search_args['domain_id'] = data['domain'].id
    data['role'] = self.identity.find_role(name_or_id=role)
    if not data['role']:
        raise exceptions.SDKException('Role {0} not found.'.format(role))
    if user:
        data['user'] = self.get_user(user, filters=search_args)
    if group:
        data['group'] = self.identity.find_group(group, ignore_missing=False, **search_args)
    if data.get('user') and data.get('group'):
        raise exceptions.SDKException('Specify either a group or a user, not both')
    if data.get('user') is None and data.get('group') is None:
        raise exceptions.SDKException('Must specify either a user or a group')
    if project is None and domain is None and (system is None):
        raise exceptions.SDKException('Must specify either a domain, project or system')
    if project:
        data['project'] = self.identity.find_project(project, ignore_missing=False, **search_args)
    return data