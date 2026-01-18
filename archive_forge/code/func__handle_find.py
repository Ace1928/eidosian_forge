from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
def _handle_find(self):
    """
        Handles the Ansible task when the command is to find an entity
        """
    entities = self._find_entities(entity_id=self.entity_id, entity_class=self.entity_class, match_filter=self.match_filter, properties=self.properties, entity_fetcher=self.entity_fetcher)
    self.result['changed'] = False
    if entities:
        if len(entities) == 1:
            self.result['id'] = entities[0].id
        for entity in entities:
            self.result['entities'].append(entity.to_dict())
    elif not self.module.check_mode:
        self.module.fail_json(msg='Unable to find matching entries')