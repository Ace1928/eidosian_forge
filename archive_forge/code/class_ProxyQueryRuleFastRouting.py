from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.proxysql.plugins.module_utils.mysql import (
from ansible.module_utils.six import iteritems
from ansible.module_utils._text import to_native
class ProxyQueryRuleFastRouting(object):

    def __init__(self, module):
        self.state = module.params['state']
        self.force_delete = module.params['force_delete']
        self.save_to_disk = module.params['save_to_disk']
        self.load_to_runtime = module.params['load_to_runtime']
        config_data_keys = ['username', 'schemaname', 'flagIN', 'destination_hostgroup', 'comment']
        self.config_data = dict(((k, module.params[k]) for k in config_data_keys))

    def check_rule_pk_exists(self, cursor):
        query_string = 'SELECT count(*) AS `rule_count` FROM mysql_query_rules_fast_routing WHERE username = %s  AND schemaname = %s AND flagIN = %s'
        query_data = [self.config_data['username'], self.config_data['schemaname'], self.config_data['flagIN']]
        cursor.execute(query_string, query_data)
        check_count = cursor.fetchone()
        return int(check_count['rule_count']) > 0

    def check_rule_cfg_exists(self, cursor):
        query_string = 'SELECT count(*) AS `rule_count` FROM mysql_query_rules_fast_routing'
        cols = 0
        query_data = []
        for col, val in iteritems(self.config_data):
            if val is not None:
                cols += 1
                query_data.append(val)
                if cols == 1:
                    query_string += ' WHERE ' + col + ' = %s'
                else:
                    query_string += ' AND ' + col + ' = %s'
        if cols > 0:
            cursor.execute(query_string, query_data)
        else:
            cursor.execute(query_string)
        check_count = cursor.fetchone()
        return int(check_count['rule_count'])

    def get_rule_config(self, cursor):
        query_string = 'SELECT * FROM mysql_query_rules_fast_routing WHERE username = %s AND schemaname = %s AND flagIN = %s'
        query_data = [self.config_data['username'], self.config_data['schemaname'], self.config_data['flagIN']]
        for col, val in iteritems(self.config_data):
            if val is not None:
                query_data.append(val)
                query_string += ' AND ' + col + ' = %s'
        cursor.execute(query_string, query_data)
        rule = cursor.fetchall()
        return rule

    def create_rule_config(self, cursor):
        query_string = 'INSERT INTO mysql_query_rules_fast_routing ('
        cols = 0
        query_data = []
        for col, val in iteritems(self.config_data):
            if val is not None:
                cols += 1
                query_data.append(val)
                query_string += col + ','
        query_string = query_string[:-1]
        query_string += ') VALUES (' + '%s, ' * cols
        query_string = query_string[:-2]
        query_string += ')'
        cursor.execute(query_string, query_data)
        return True

    def update_rule_config(self, cursor):
        query_string = 'UPDATE mysql_query_rules_fast_routing'
        cols = 0
        query_data = [self.config_data['username'], self.config_data['schemaname'], self.config_data['flagIN']]
        for col, val in iteritems(self.config_data):
            if val is not None and col not in ('username', 'schemaname', 'flagIN'):
                query_data.insert(cols, val)
                cols += 1
                if cols == 1:
                    query_string += ' SET ' + col + '= %s,'
                else:
                    query_string += ' ' + col + ' = %s,'
        query_string = query_string[:-1]
        query_string += 'WHERE username = %s AND schemaname = %s AND flagIN = %s'
        cursor.execute(query_string, query_data)
        return True

    def delete_rule_config(self, cursor):
        query_string = 'DELETE FROM mysql_query_rules_fast_routing'
        cols = 0
        query_data = []
        for col, val in iteritems(self.config_data):
            if val is not None:
                cols += 1
                query_data.append(val)
                if cols == 1:
                    query_string += ' WHERE ' + col + ' = %s'
                else:
                    query_string += ' AND ' + col + ' = %s'
        if cols > 0:
            cursor.execute(query_string, query_data)
        else:
            cursor.execute(query_string)
        check_count = cursor.rowcount
        return (True, int(check_count))

    def manage_config(self, cursor, changed):
        if not changed:
            return
        if self.save_to_disk:
            save_config_to_disk(cursor, 'QUERY RULES')
        if self.load_to_runtime:
            load_config_to_runtime(cursor, 'QUERY RULES')

    def create_rule(self, check_mode, result, cursor):
        if not check_mode:
            result['changed'] = self.create_rule_config(cursor)
            result['msg'] = 'Added rule to mysql_query_rules_fast_routing.'
            self.manage_config(cursor, result['changed'])
            result['rules'] = self.get_rule_config(cursor)
        else:
            result['changed'] = True
            result['msg'] = 'Rule would have been added to mysql_query_rules_fast_routing, however check_mode is enabled.'

    def update_rule(self, check_mode, result, cursor):
        if not check_mode:
            result['changed'] = self.update_rule_config(cursor)
            result['msg'] = 'Updated rule in mysql_query_rules_fast_routing.'
            self.manage_config(cursor, result['changed'])
            result['rules'] = self.get_rule_config(cursor)
        else:
            result['changed'] = True
            result['msg'] = 'Rule would have been updated in mysql_query_rules_fast_routing, however check_mode is enabled.'

    def delete_rule(self, check_mode, result, cursor):
        if not check_mode:
            result['rules'] = self.get_rule_config(cursor)
            result['changed'], result['rows_affected'] = self.delete_rule_config(cursor)
            result['msg'] = 'Deleted rule from mysql_query_rules_fast_routing.'
            self.manage_config(cursor, result['changed'])
        else:
            result['changed'] = True
            result['msg'] = 'Rule would have been deleted from mysql_query_rules_fast_routing, however check_mode is enabled.'