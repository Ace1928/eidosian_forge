from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.postgresql.plugins.module_utils.database import \
from ansible_collections.community.postgresql.plugins.module_utils.postgres import (
def __exists_in_db(self):
    """Check index existence, collect info, add it to self.info dict.

        Return True if the index exists, otherwise, return False.
        """
    query = 'SELECT i.schemaname, i.tablename, i.tablespace, pi.indisvalid, c.reloptions FROM pg_catalog.pg_indexes AS i JOIN pg_catalog.pg_class AS c ON i.indexname = c.relname JOIN pg_catalog.pg_index AS pi ON c.oid = pi.indexrelid WHERE i.indexname = %(name)s'
    res = exec_sql(self, query, query_params={'name': self.name}, add_to_executed=False)
    if res:
        self.exists = True
        self.info = dict(name=self.name, state='present', schema=res[0]['schemaname'], tblname=res[0]['tablename'], tblspace=res[0]['tablespace'] if res[0]['tablespace'] else '', valid=res[0]['indisvalid'], storage_params=res[0]['reloptions'] if res[0]['reloptions'] else [])
        return True
    else:
        self.exists = False
        return False