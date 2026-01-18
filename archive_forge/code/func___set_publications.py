from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems
from ansible_collections.community.postgresql.plugins.module_utils.database import \
from ansible_collections.community.postgresql.plugins.module_utils.postgres import (
def __set_publications(self, publications, check_mode=True):
    """Update publications.

        Args:
            publications (list): Publications on the primary to use.

        Kwargs:
            check_mode (bool): If True, don't actually change anything,
                just make SQL, add it to ``self.executed_queries`` and return True.

        Returns:
            True if successful, False otherwise.
        """
    query = 'ALTER SUBSCRIPTION %s SET PUBLICATION %s' % (self.name, ', '.join(publications))
    return self.__exec_sql(query, check_mode=check_mode)