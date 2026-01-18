from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.postgresql.plugins.module_utils.database import (
from ansible_collections.community.postgresql.plugins.module_utils.postgres import (
def __set_seq_owner(self):
    """Set the sequence owner."""
    query = 'ALTER SEQUENCE %s OWNER TO "%s"' % (pg_quote_identifier(self.obj_name, 'sequence'), self.role)
    self.changed = exec_sql(self, query, return_bool=True)