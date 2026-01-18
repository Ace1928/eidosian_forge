from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
def _unassign_member(self, entity_fetcher, entity, entity_class, parent, set_output):
    """
        Removes the entity as a member of a parent
        :param entity_fetcher: The fetcher of the entity type
        :param entity: The entity to remove as a member
        :param entity_class: The class of the entity
        :param parent: The parent on which to add the entity as a member
        :param set_output: If set to True, sets the Ansible result variables
        """
    members = []
    for member in entity_fetcher.get():
        if member.id != entity.id:
            members.append(member)
    try:
        parent.assign(members, entity_class)
    except BambouHTTPError as error:
        self.module.fail_json(msg='Unable to remove entity as a member: {0}'.format(error))
    self.result['changed'] = True
    if set_output:
        self.result['id'] = entity.id
        self.result['entities'].append(entity.to_dict())