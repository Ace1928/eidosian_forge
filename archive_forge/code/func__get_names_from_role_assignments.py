import copy
import itertools
from oslo_log import log
from keystone.common import cache
from keystone.common import driver_hints
from keystone.common import manager
from keystone.common import provider_api
from keystone.common.resource_options import options as ro_opt
import keystone.conf
from keystone import exception
from keystone.i18n import _
from keystone import notifications
def _get_names_from_role_assignments(self, role_assignments):
    role_assign_list = []
    for role_asgmt in role_assignments:
        new_assign = copy.deepcopy(role_asgmt)
        for key, value in role_asgmt.items():
            if key == 'domain_id':
                _domain = PROVIDERS.resource_api.get_domain(value)
                new_assign['domain_name'] = _domain['name']
            elif key == 'user_id':
                try:
                    _user = PROVIDERS.identity_api.get_user(value)
                except exception.UserNotFound:
                    msg = 'User %(user)s not found in the backend but still has role assignments.'
                    LOG.warning(msg, {'user': value})
                    new_assign['user_name'] = ''
                    new_assign['user_domain_id'] = ''
                    new_assign['user_domain_name'] = ''
                else:
                    new_assign['user_name'] = _user['name']
                    new_assign['user_domain_id'] = _user['domain_id']
                    new_assign['user_domain_name'] = PROVIDERS.resource_api.get_domain(_user['domain_id'])['name']
            elif key == 'group_id':
                try:
                    _group = PROVIDERS.identity_api.get_group(value)
                except exception.GroupNotFound:
                    msg = 'Group %(group)s not found in the backend but still has role assignments.'
                    LOG.warning(msg, {'group': value})
                    new_assign['group_name'] = ''
                    new_assign['group_domain_id'] = ''
                    new_assign['group_domain_name'] = ''
                else:
                    new_assign['group_name'] = _group['name']
                    new_assign['group_domain_id'] = _group['domain_id']
                    new_assign['group_domain_name'] = PROVIDERS.resource_api.get_domain(_group['domain_id'])['name']
            elif key == 'project_id':
                _project = PROVIDERS.resource_api.get_project(value)
                new_assign['project_name'] = _project['name']
                new_assign['project_domain_id'] = _project['domain_id']
                new_assign['project_domain_name'] = PROVIDERS.resource_api.get_domain(_project['domain_id'])['name']
            elif key == 'role_id':
                _role = PROVIDERS.role_api.get_role(value)
                new_assign['role_name'] = _role['name']
                if _role['domain_id'] is not None:
                    new_assign['role_domain_id'] = _role['domain_id']
                    new_assign['role_domain_name'] = PROVIDERS.resource_api.get_domain(_role['domain_id'])['name']
        role_assign_list.append(new_assign)
    return role_assign_list