from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems
from ansible_collections.community.postgresql.plugins.module_utils.database import (
from ansible_collections.community.postgresql.plugins.module_utils.postgres import (
def __pub_set_tables(self, tables, check_mode=False):
    """Set a table suit that need to be published by the publication.

        Args:
            tables (list): List of tables.

        Kwargs:
            check_mode (bool): If True, don't actually change anything,
                just make SQL, add it to ``self.executed_queries`` and return True.

        Returns:
            True if successful, False otherwise.
        """
    quoted_tables = [pg_quote_identifier(t, 'table') for t in tables]
    query = 'ALTER PUBLICATION %s SET TABLE %s' % (pg_quote_identifier(self.name, 'publication'), ', '.join(quoted_tables))
    return self.__exec_sql(query, check_mode=check_mode)