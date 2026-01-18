from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
def _find_parent(self):
    """
        Fetches the parent if needed, otherwise configures the root object as parent. Also configures the entity fetcher
        Important notes:
        - If the parent is not set, the parent is automatically set to the root object
        - It the root object does not hold a fetcher for the entity, you have to provide an ID
        - If you want to assign/unassign, you have to provide a valid parent
        """
    self.parent = self.nuage_connection.user
    if self.parent_id:
        self.parent = self.parent_class(id=self.parent_id)
        try:
            self.parent.fetch()
        except BambouHTTPError as error:
            self.module.fail_json(msg='Failed to fetch the specified parent: {0}'.format(error))
    self.entity_fetcher = self.parent.fetcher_for_rest_name(self.entity_class.rest_name)