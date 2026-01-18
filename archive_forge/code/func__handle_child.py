from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
def _handle_child(self, child, parent):
    """
        Handles children of a main entity. Fields are similar to the normal fields
        Currently only supported state: present
        """
    if 'type' not in list(child.keys()):
        self.module.fail_json(msg='Child type unspecified')
    elif 'id' not in list(child.keys()) and 'properties' not in list(child.keys()):
        self.module.fail_json(msg='Child ID or properties unspecified')
    child_id = None
    if 'id' in list(child.keys()):
        child_id = child['id']
    child_properties = None
    if 'properties' in list(child.keys()):
        child_properties = child['properties']
    child_filter = None
    if 'match_filter' in list(child.keys()):
        child_filter = child['match_filter']
    entity_class = None
    try:
        entity_class = getattr(VSPK, 'NU{0:s}'.format(child['type']))
    except AttributeError:
        self.module.fail_json(msg='Unrecognised child type specified')
    entity_fetcher = parent.fetcher_for_rest_name(entity_class.rest_name)
    if entity_fetcher is None and (not child_id) and (not self.module.check_mode):
        self.module.fail_json(msg='Unable to find a fetcher for child, and no ID specified.')
    entity = self._find_entity(entity_id=child_id, entity_class=entity_class, match_filter=child_filter, properties=child_properties, entity_fetcher=entity_fetcher)
    if entity_fetcher.relationship == 'member' and (not entity):
        self.module.fail_json(msg='Trying to assign a child that does not exist')
    elif entity_fetcher.relationship == 'member' and entity:
        if not self._is_member(entity_fetcher=entity_fetcher, entity=entity):
            if self.module.check_mode:
                self.result['changed'] = True
            else:
                self._assign_member(entity_fetcher=entity_fetcher, entity=entity, entity_class=entity_class, parent=parent, set_output=False)
    elif entity_fetcher.relationship in ['child', 'root'] and (not entity):
        if self.module.check_mode:
            self.result['changed'] = True
        else:
            entity = self._create_entity(entity_class=entity_class, parent=parent, properties=child_properties)
    elif entity_fetcher.relationship in ['child', 'root'] and entity:
        changed = self._has_changed(entity=entity, properties=child_properties)
        if self.module.check_mode:
            self.result['changed'] = changed
        elif changed:
            entity = self._save_entity(entity=entity)
    if entity:
        self.result['entities'].append(entity.to_dict())
    if 'children' in list(child.keys()) and (not self.module.check_mode):
        for subchild in child['children']:
            self._handle_child(child=subchild, parent=entity)