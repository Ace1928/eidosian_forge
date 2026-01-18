from __future__ import (absolute_import, division, print_function)
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ovirt.ovirt.plugins.module_utils.ovirt import (
def _get_sched_policy(self):
    sched_policy = None
    if self.param('scheduling_policy'):
        sched_policies_service = self._connection.system_service().scheduling_policies_service()
        sched_policy = search_by_name(sched_policies_service, self.param('scheduling_policy'))
        if not sched_policy:
            raise Exception("Scheduling policy '%s' was not found" % self.param('scheduling_policy'))
    return sched_policy