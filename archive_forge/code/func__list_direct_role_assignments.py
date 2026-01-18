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
def _list_direct_role_assignments(self, role_id, user_id, group_id, system, domain_id, project_id, subtree_ids, inherited):
    """List role assignments without applying expansion.

        Returns a list of direct role assignments, where their attributes match
        the provided filters. If subtree_ids is not None, then we also want to
        include all subtree_ids in the filter as well.

        """
    group_ids = [group_id] if group_id else None
    project_ids_of_interest = None
    if project_id:
        if subtree_ids:
            project_ids_of_interest = subtree_ids + [project_id]
        else:
            project_ids_of_interest = [project_id]
    project_and_domain_assignments = []
    if not system:
        project_and_domain_assignments = self.driver.list_role_assignments(role_id=role_id, user_id=user_id, group_ids=group_ids, domain_id=domain_id, project_ids=project_ids_of_interest, inherited_to_projects=inherited)
    system_assignments = []
    if system or (not project_id and (not domain_id) and (not system)):
        if user_id:
            assignments = self.list_system_grants_for_user(user_id)
            for assignment in assignments:
                system_assignments.append({'system': {'all': True}, 'user_id': user_id, 'role_id': assignment['id']})
        elif group_id:
            assignments = self.list_system_grants_for_group(group_id)
            for assignment in assignments:
                system_assignments.append({'system': {'all': True}, 'group_id': group_id, 'role_id': assignment['id']})
        else:
            assignments = self.list_all_system_grants()
            for assignment in assignments:
                a = {}
                if assignment['type'] == self._GROUP_SYSTEM:
                    a['group_id'] = assignment['actor_id']
                elif assignment['type'] == self._USER_SYSTEM:
                    a['user_id'] = assignment['actor_id']
                a['role_id'] = assignment['role_id']
                a['system'] = {'all': True}
                system_assignments.append(a)
        if role_id:
            system_assignments = [sa for sa in system_assignments if role_id == sa['role_id']]
    assignments = []
    for assignment in itertools.chain(project_and_domain_assignments, system_assignments):
        assignments.append(assignment)
    return assignments