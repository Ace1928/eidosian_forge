from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.proxysql.plugins.module_utils.mysql import (
from ansible.module_utils.six import iteritems
from ansible.module_utils._text import to_native, to_bytes
from hashlib import sha1
class ProxySQLUser(object):

    def __init__(self, module):
        self.state = module.params['state']
        self.save_to_disk = module.params['save_to_disk']
        self.load_to_runtime = module.params['load_to_runtime']
        self.username = module.params['username']
        self.backend = module.params['backend']
        self.frontend = module.params['frontend']
        config_data_keys = ['password', 'active', 'use_ssl', 'default_hostgroup', 'default_schema', 'transaction_persistent', 'fast_forward', 'max_connections']
        self.config_data = dict(((k, module.params[k]) for k in config_data_keys))
        if module.params['password'] is not None and module.params['encrypt_password']:
            encryption_method = encryption_method_map[module.params['encryption_method']]
            encrypted_password = encrypt_cleartext_password(module.params['password'], encryption_method)
            self.config_data['password'] = encrypted_password

    def check_user_config_exists(self, cursor):
        query_string = 'SELECT count(*) AS `user_count`\n               FROM mysql_users\n               WHERE username = %s\n                 AND backend = %s\n                 AND frontend = %s'
        query_data = [self.username, self.backend, self.frontend]
        cursor.execute(query_string, query_data)
        check_count = cursor.fetchone()
        return int(check_count['user_count']) > 0

    def check_user_privs(self, cursor):
        query_string = 'SELECT count(*) AS `user_count`\n               FROM mysql_users\n               WHERE username = %s\n                 AND backend = %s\n                 AND frontend = %s'
        query_data = [self.username, self.backend, self.frontend]
        for col, val in iteritems(self.config_data):
            if val is not None:
                query_data.append(val)
                query_string += '\n  AND ' + col + ' = %s'
        cursor.execute(query_string, query_data)
        check_count = cursor.fetchone()
        return int(check_count['user_count']) > 0

    def get_user_config(self, cursor):
        query_string = 'SELECT *\n               FROM mysql_users\n               WHERE username = %s\n                 AND backend = %s\n                 AND frontend = %s'
        query_data = [self.username, self.backend, self.frontend]
        cursor.execute(query_string, query_data)
        user = cursor.fetchone()
        return user

    def create_user_config(self, cursor):
        query_string = 'INSERT INTO mysql_users (\n               username,\n               backend,\n               frontend'
        cols = 3
        query_data = [self.username, self.backend, self.frontend]
        for col, val in iteritems(self.config_data):
            if val is not None:
                cols += 1
                query_data.append(val)
                query_string += ',\n' + col
        query_string += ')\n' + 'VALUES (' + '%s ,' * cols
        query_string = query_string[:-2]
        query_string += ')'
        cursor.execute(query_string, query_data)
        return True

    def update_user_config(self, cursor):
        query_string = 'UPDATE mysql_users'
        cols = 0
        query_data = []
        for col, val in iteritems(self.config_data):
            if val is not None:
                cols += 1
                query_data.append(val)
                if cols == 1:
                    query_string += '\nSET ' + col + '= %s,'
                else:
                    query_string += '\n    ' + col + ' = %s,'
        query_string = query_string[:-1]
        query_string += '\nWHERE username = %s\n  AND backend = %s' + '\n  AND frontend = %s'
        query_data.append(self.username)
        query_data.append(self.backend)
        query_data.append(self.frontend)
        cursor.execute(query_string, query_data)
        return True

    def delete_user_config(self, cursor):
        query_string = 'DELETE FROM mysql_users\n               WHERE username = %s\n                 AND backend = %s\n                 AND frontend = %s'
        query_data = [self.username, self.backend, self.frontend]
        cursor.execute(query_string, query_data)
        return True

    def manage_config(self, cursor, state):
        if state:
            if self.save_to_disk:
                save_config_to_disk(cursor, 'USERS')
            if self.load_to_runtime:
                load_config_to_runtime(cursor, 'USERS')

    def create_user(self, check_mode, result, cursor):
        if not check_mode:
            result['changed'] = self.create_user_config(cursor)
            result['msg'] = 'Added user to mysql_users'
            result['user'] = self.get_user_config(cursor)
            self.manage_config(cursor, result['changed'])
        else:
            result['changed'] = True
            result['msg'] = 'User would have been added to' + ' mysql_users, however check_mode' + ' is enabled.'

    def update_user(self, check_mode, result, cursor):
        if not check_mode:
            result['changed'] = self.update_user_config(cursor)
            result['msg'] = 'Updated user in mysql_users'
            result['user'] = self.get_user_config(cursor)
            self.manage_config(cursor, result['changed'])
        else:
            result['changed'] = True
            result['msg'] = 'User would have been updated in' + ' mysql_users, however check_mode' + ' is enabled.'

    def delete_user(self, check_mode, result, cursor):
        if not check_mode:
            result['user'] = self.get_user_config(cursor)
            result['changed'] = self.delete_user_config(cursor)
            result['msg'] = 'Deleted user from mysql_users'
            self.manage_config(cursor, result['changed'])
        else:
            result['changed'] = True
            result['msg'] = 'User would have been deleted from' + ' mysql_users, however check_mode is' + ' enabled.'