from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.mysql.plugins.module_utils.mysql import (
from ansible_collections.community.mysql.plugins.module_utils.user import (
from ansible.module_utils._text import to_native
from ansible.module_utils.six import iteritems
def __get_members(self):
    """Get current role's members.

        Returns:
            set: Members.
        """
    if self.is_mariadb:
        self.cursor.execute('select user, host from mysql.roles_mapping where role = %s', (self.name,))
    else:
        self.cursor.execute('select TO_USER as user, TO_HOST as host from mysql.role_edges where FROM_USER = %s', (self.name,))
    return set(self.cursor.fetchall())