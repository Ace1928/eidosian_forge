from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.postgresql.plugins.module_utils.database import \
from ansible_collections.community.postgresql.plugins.module_utils.postgres import (
class PgSlot(object):

    def __init__(self, module, cursor, name):
        self.module = module
        self.cursor = cursor
        self.name = name
        self.exists = False
        self.kind = ''
        self.__slot_exists()
        self.changed = False
        self.executed_queries = []

    def create(self, kind='physical', immediately_reserve=False, output_plugin=False, just_check=False):
        if self.exists:
            if self.kind == kind:
                return False
            else:
                self.module.warn("slot with name '%s' already exists but has another type '%s'" % (self.name, self.kind))
                return False
        if just_check:
            return None
        if kind == 'physical':
            if get_server_version(self.cursor.connection) < 90600:
                query = 'SELECT pg_create_physical_replication_slot(%(name)s)'
            else:
                query = 'SELECT pg_create_physical_replication_slot(%(name)s, %(i_reserve)s)'
            self.changed = exec_sql(self, query, query_params={'name': self.name, 'i_reserve': immediately_reserve}, return_bool=True)
        elif kind == 'logical':
            query = 'SELECT pg_create_logical_replication_slot(%(name)s, %(o_plugin)s)'
            self.changed = exec_sql(self, query, query_params={'name': self.name, 'o_plugin': output_plugin}, return_bool=True)

    def drop(self):
        if not self.exists:
            return False
        query = 'SELECT pg_drop_replication_slot(%(name)s)'
        self.changed = exec_sql(self, query, query_params={'name': self.name}, return_bool=True)

    def __slot_exists(self):
        query = 'SELECT slot_type FROM pg_replication_slots WHERE slot_name = %(name)s'
        res = exec_sql(self, query, query_params={'name': self.name}, add_to_executed=False)
        if res:
            self.exists = True
            self.kind = res[0]['slot_type']