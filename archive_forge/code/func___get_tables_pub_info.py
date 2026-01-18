from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems
from ansible_collections.community.postgresql.plugins.module_utils.database import (
from ansible_collections.community.postgresql.plugins.module_utils.postgres import (
def __get_tables_pub_info(self):
    """Get and return tables that are published by the publication.

        Returns:
            List of dicts with published tables.
        """
    query = "SELECT schemaname || '.' || tablename as schema_dot_table FROM pg_publication_tables WHERE pubname = %(pname)s"
    return exec_sql(self, query, query_params={'pname': self.name}, add_to_executed=False)