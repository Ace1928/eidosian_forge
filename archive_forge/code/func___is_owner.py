from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.postgresql.plugins.module_utils.database import (
from ansible_collections.community.postgresql.plugins.module_utils.postgres import (
def __is_owner(self):
    """Return True if self.role is the current object owner."""
    if self.obj_type == 'table':
        query = 'SELECT 1 FROM pg_tables WHERE tablename = %(obj_name)s AND tableowner = %(role)s'
    elif self.obj_type == 'database':
        query = 'SELECT 1 FROM pg_database AS d JOIN pg_roles AS r ON d.datdba = r.oid WHERE d.datname = %(obj_name)s AND r.rolname = %(role)s'
    elif self.obj_type in ('aggregate', 'function', 'routine', 'procedure'):
        if self.obj_type == 'routine' and self.pg_version < 110000:
            self.module.fail_json(msg='PostgreSQL version must be >= 11 for obj_type=routine.')
        if self.obj_type == 'procedure' and self.pg_version < 110000:
            self.module.fail_json(msg='PostgreSQL version must be >= 11 for obj_type=procedure.')
        query = 'SELECT 1 FROM pg_proc AS f JOIN pg_roles AS r ON f.proowner = r.oid WHERE f.proname = %(obj_name)s AND r.rolname = %(role)s'
    elif self.obj_type == 'sequence':
        query = "SELECT 1 FROM pg_class AS c JOIN pg_roles AS r ON c.relowner = r.oid WHERE c.relkind = 'S' AND c.relname = %(obj_name)s AND r.rolname = %(role)s"
    elif self.obj_type == 'schema':
        query = 'SELECT 1 FROM information_schema.schemata WHERE schema_name = %(obj_name)s AND schema_owner = %(role)s'
    elif self.obj_type == 'tablespace':
        query = 'SELECT 1 FROM pg_tablespace AS t JOIN pg_roles AS r ON t.spcowner = r.oid WHERE t.spcname = %(obj_name)s AND r.rolname = %(role)s'
    elif self.obj_type == 'view':
        query = 'SELECT 1 FROM pg_views WHERE viewname = %(obj_name)s AND viewowner = %(role)s'
    elif self.obj_type == 'matview':
        if self.pg_version < 90300:
            self.module.fail_json(msg='PostgreSQL version must be >= 9.3 for obj_type=matview.')
        query = 'SELECT 1 FROM pg_matviews WHERE matviewname = %(obj_name)s AND matviewowner = %(role)s'
    elif self.obj_type in ('domain', 'type'):
        query = 'SELECT 1 FROM pg_type AS t JOIN pg_roles AS r ON t.typowner = r.oid WHERE t.typname = %(obj_name)s AND r.rolname = %(role)s'
    elif self.obj_type == 'language':
        query = 'SELECT 1 FROM pg_language AS l JOIN pg_roles AS r ON l.lanowner = r.oid WHERE l.lanname = %(obj_name)s AND r.rolname = %(role)s'
    elif self.obj_type == 'collation':
        query = 'SELECT 1 FROM pg_collation AS c JOIN pg_roles AS r ON c.collowner = r.oid WHERE c.collname = %(obj_name)s AND r.rolname = %(role)s'
    elif self.obj_type == 'conversion':
        query = 'SELECT 1 FROM pg_conversion AS c JOIN pg_roles AS r ON c.conowner = r.oid WHERE c.conname = %(obj_name)s AND r.rolname = %(role)s'
    elif self.obj_type == 'text_search_configuration':
        query = 'SELECT 1 FROM pg_ts_config AS t JOIN pg_roles AS r ON t.cfgowner = r.oid WHERE t.cfgname = %(obj_name)s AND r.rolname = %(role)s'
    elif self.obj_type == 'text_search_dictionary':
        query = 'SELECT 1 FROM pg_ts_dict AS t JOIN pg_roles AS r ON t.dictowner = r.oid WHERE t.dictname = %(obj_name)s AND r.rolname = %(role)s'
    elif self.obj_type == 'foreign_data_wrapper':
        query = 'SELECT 1 FROM pg_foreign_data_wrapper AS f JOIN pg_roles AS r ON f.fdwowner = r.oid WHERE f.fdwname = %(obj_name)s AND r.rolname = %(role)s'
    elif self.obj_type == 'server':
        query = 'SELECT 1 FROM pg_foreign_server AS f JOIN pg_roles AS r ON f.srvowner = r.oid WHERE f.srvname = %(obj_name)s AND r.rolname = %(role)s'
    elif self.obj_type == 'foreign_table':
        query = "SELECT 1 FROM pg_class AS c JOIN pg_roles AS r ON c.relowner = r.oid WHERE c.relkind = 'f' AND c.relname = %(obj_name)s AND r.rolname = %(role)s"
    elif self.obj_type == 'event_trigger':
        if self.pg_version < 110000:
            self.module.fail_json(msg='PostgreSQL version must be >= 11 for obj_type=event_trigger.')
        query = 'SELECT 1 FROM pg_event_trigger AS e JOIN pg_roles AS r ON e.evtowner = r.oid WHERE e.evtname = %(obj_name)s AND r.rolname = %(role)s'
    elif self.obj_type == 'large_object':
        query = 'SELECT 1 FROM pg_largeobject_metadata AS l JOIN pg_roles AS r ON l.lomowner = r.oid WHERE l.oid = %(obj_name)s AND r.rolname = %(role)s'
    elif self.obj_type == 'publication':
        if self.pg_version < 110000:
            self.module.fail_json(msg='PostgreSQL version must be >= 11 for obj_type=publication.')
        query = 'SELECT 1 FROM pg_publication AS p JOIN pg_roles AS r ON p.pubowner = r.oid WHERE p.pubname = %(obj_name)s AND r.rolname = %(role)s'
    elif self.obj_type == 'statistics':
        if self.pg_version < 110000:
            self.module.fail_json(msg='PostgreSQL version must be >= 11 for obj_type=statistics.')
        query = 'SELECT 1 FROM pg_statistic_ext AS s JOIN pg_roles AS r ON s.stxowner = r.oid WHERE s.stxname = %(obj_name)s AND r.rolname = %(role)s'
    if self.obj_type in ('function', 'aggregate', 'procedure', 'routine'):
        query_params = {'obj_name': self.obj_name.split('(')[0], 'role': self.role}
    else:
        query_params = {'obj_name': self.obj_name, 'role': self.role}
    return exec_sql(self, query, query_params, add_to_executed=False)