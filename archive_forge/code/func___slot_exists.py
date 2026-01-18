from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.postgresql.plugins.module_utils.database import \
from ansible_collections.community.postgresql.plugins.module_utils.postgres import (
def __slot_exists(self):
    query = 'SELECT slot_type FROM pg_replication_slots WHERE slot_name = %(name)s'
    res = exec_sql(self, query, query_params={'name': self.name}, add_to_executed=False)
    if res:
        self.exists = True
        self.kind = res[0]['slot_type']