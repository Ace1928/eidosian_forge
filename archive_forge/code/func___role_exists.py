from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.mysql.plugins.module_utils.mysql import (
from ansible_collections.community.mysql.plugins.module_utils.user import (
from ansible.module_utils._text import to_native
from ansible.module_utils.six import iteritems
def __role_exists(self):
    """Check if a role exists.

        Returns:
            bool: True if the role exists, False if it does not.
        """
    self.cursor.execute(*self.q_builder.role_exists())
    return self.cursor.fetchone()[0] > 0