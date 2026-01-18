from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.proxysql.plugins.module_utils.mysql import (
from ansible.module_utils._text import to_native
class ProxySQLReplicationHostgroup(object):

    def __init__(self, module, version):
        self.state = module.params['state']
        self.save_to_disk = module.params['save_to_disk']
        self.load_to_runtime = module.params['load_to_runtime']
        self.writer_hostgroup = module.params['writer_hostgroup']
        self.reader_hostgroup = module.params['reader_hostgroup']
        self.comment = module.params['comment']
        self.check_type = module.params['check_type']
        self.check_type_support = version.get('major') >= 2
        self.check_mode = module.check_mode

    def check_repl_group_config(self, cursor, keys):
        query_string = 'SELECT count(*) AS `repl_groups`\n               FROM mysql_replication_hostgroups\n               WHERE writer_hostgroup = %s'
        query_data = [self.writer_hostgroup]
        cursor.execute(query_string, query_data)
        check_count = cursor.fetchone()
        return int(check_count['repl_groups']) > 0

    def get_repl_group_config(self, cursor):
        query_string = 'SELECT *\n               FROM mysql_replication_hostgroups\n               WHERE writer_hostgroup = %s'
        query_data = [self.writer_hostgroup]
        cursor.execute(query_string, query_data)
        repl_group = cursor.fetchone()
        return repl_group

    def create_repl_group_config(self, cursor):
        query_string = 'INSERT INTO mysql_replication_hostgroups (\n               writer_hostgroup,\n               reader_hostgroup,\n               comment)\n               VALUES (%s, %s, %s)'
        query_data = [self.writer_hostgroup, self.reader_hostgroup, self.comment or '']
        cursor.execute(query_string, query_data)
        if self.check_type_support:
            self.update_check_type(cursor)
        return True

    def delete_repl_group_config(self, cursor):
        query_string = 'DELETE FROM mysql_replication_hostgroups\n               WHERE writer_hostgroup = %s'
        query_data = [self.writer_hostgroup]
        cursor.execute(query_string, query_data)
        return True

    def manage_config(self, cursor, state):
        if state and (not self.check_mode):
            if self.save_to_disk:
                save_config_to_disk(cursor, 'SERVERS')
            if self.load_to_runtime:
                load_config_to_runtime(cursor, 'SERVERS')

    def create_repl_group(self, result, cursor):
        if not self.check_mode:
            result['changed'] = self.create_repl_group_config(cursor)
            result['msg'] = 'Added server to mysql_hosts'
            result['repl_group'] = self.get_repl_group_config(cursor)
            self.manage_config(cursor, result['changed'])
        else:
            result['changed'] = True
            result['msg'] = 'Repl group would have been added to' + ' mysql_replication_hostgroups, however' + ' check_mode is enabled.'

    def update_repl_group(self, result, cursor):
        current = self.get_repl_group_config(cursor)
        if self.check_type_support and current.get('check_type') != self.check_type:
            result['changed'] = True
            if not self.check_mode:
                result['msg'] = 'Updated replication hostgroups'
                self.update_check_type(cursor)
            else:
                result['msg'] = 'Updated replication hostgroups in check_mode'
        if current.get('comment') != self.comment:
            result['changed'] = True
            result['msg'] = 'Updated replication hostgroups in check_mode'
            if not self.check_mode:
                result['msg'] = 'Updated replication hostgroups'
                self.update_comment(cursor)
        if int(current.get('reader_hostgroup')) != self.reader_hostgroup:
            result['changed'] = True
            result['msg'] = 'Updated replication hostgroups in check_mode'
            if not self.check_mode:
                result['msg'] = 'Updated replication hostgroups'
                self.update_reader_hostgroup(cursor)
        result['repl_group'] = self.get_repl_group_config(cursor)
        self.manage_config(cursor, result['changed'])

    def delete_repl_group(self, result, cursor):
        if not self.check_mode:
            result['repl_group'] = self.get_repl_group_config(cursor)
            result['changed'] = self.delete_repl_group_config(cursor)
            result['msg'] = 'Deleted server from mysql_hosts'
            self.manage_config(cursor, result['changed'])
        else:
            result['changed'] = True
            result['msg'] = 'Repl group would have been deleted from' + ' mysql_replication_hostgroups, however' + ' check_mode is enabled.'

    def update_check_type(self, cursor):
        try:
            query_string = 'UPDATE mysql_replication_hostgroups SET check_type = %s WHERE writer_hostgroup = %s'
            cursor.execute(query_string, (self.check_type, self.writer_hostgroup))
        except Exception as e:
            pass

    def update_reader_hostgroup(self, cursor):
        query_string = 'UPDATE mysql_replication_hostgroups SET reader_hostgroup = %s WHERE writer_hostgroup = %s'
        cursor.execute(query_string, (self.reader_hostgroup, self.writer_hostgroup))

    def update_comment(self, cursor):
        query_string = 'UPDATE mysql_replication_hostgroups SET comment = %s WHERE writer_hostgroup = %s '
        cursor.execute(query_string, (self.comment, self.writer_hostgroup))