from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
def _handle_wait_for_job(self):
    """
        Handles the Ansible task when the command is to wait for a job
        """
    self.entity = self._find_entity(entity_id=self.entity_id, entity_class=self.entity_class, match_filter=self.match_filter, properties=self.properties, entity_fetcher=self.entity_fetcher)
    if self.module.check_mode:
        self.result['changed'] = True
    else:
        self._wait_for_job(self.entity)