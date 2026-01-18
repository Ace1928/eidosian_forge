from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
def _handle_absent(self):
    """
        Handles the Ansible task when the state is set to absent
        """
    self.entity = self._find_entity(entity_id=self.entity_id, entity_class=self.entity_class, match_filter=self.match_filter, properties=self.properties, entity_fetcher=self.entity_fetcher)
    if self.entity and (self.entity_fetcher is None or self.entity_fetcher.relationship in ['child', 'root']):
        if self.module.check_mode:
            self.result['changed'] = True
        else:
            self._delete_entity(self.entity)
            self.result['id'] = None
    elif self.entity and self.entity_fetcher.relationship == 'member':
        if self._is_member(entity_fetcher=self.entity_fetcher, entity=self.entity):
            if self.module.check_mode:
                self.result['changed'] = True
            else:
                self._unassign_member(entity_fetcher=self.entity_fetcher, entity=self.entity, entity_class=self.entity_class, parent=self.parent, set_output=True)