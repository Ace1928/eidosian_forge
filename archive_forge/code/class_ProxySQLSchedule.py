from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.proxysql.plugins.module_utils.mysql import (
from ansible.module_utils.six import iteritems
from ansible.module_utils._text import to_native
class ProxySQLSchedule(object):

    def __init__(self, module):
        self.state = module.params['state']
        self.force_delete = module.params['force_delete']
        self.save_to_disk = module.params['save_to_disk']
        self.load_to_runtime = module.params['load_to_runtime']
        self.active = module.params['active']
        self.interval_ms = module.params['interval_ms']
        self.filename = module.params['filename']
        config_data_keys = ['arg1', 'arg2', 'arg3', 'arg4', 'arg5', 'comment']
        self.config_data = dict(((k, module.params[k]) for k in config_data_keys))

    def check_schedule_config(self, cursor):
        query_string = 'SELECT count(*) AS `schedule_count`\n               FROM scheduler\n               WHERE active = %s\n                 AND interval_ms = %s\n                 AND filename = %s'
        query_data = [self.active, self.interval_ms, self.filename]
        for col, val in iteritems(self.config_data):
            if val is not None:
                query_data.append(val)
                query_string += '\n  AND ' + col + ' = %s'
        cursor.execute(query_string, query_data)
        check_count = cursor.fetchone()
        return int(check_count['schedule_count'])

    def get_schedule_config(self, cursor):
        query_string = 'SELECT *\n               FROM scheduler\n               WHERE active = %s\n                 AND interval_ms = %s\n                 AND filename = %s'
        query_data = [self.active, self.interval_ms, self.filename]
        for col, val in iteritems(self.config_data):
            if val is not None:
                query_data.append(val)
                query_string += '\n  AND ' + col + ' = %s'
        cursor.execute(query_string, query_data)
        schedule = cursor.fetchall()
        return schedule

    def create_schedule_config(self, cursor):
        query_string = 'INSERT INTO scheduler (\n               active,\n               interval_ms,\n               filename'
        cols = 0
        query_data = [self.active, self.interval_ms, self.filename]
        for col, val in iteritems(self.config_data):
            if val is not None:
                cols += 1
                query_data.append(val)
                query_string += ',\n' + col
        query_string += ')\n' + 'VALUES (%s, %s, %s' + ', %s' * cols + ')'
        cursor.execute(query_string, query_data)
        return True

    def delete_schedule_config(self, cursor):
        query_string = 'DELETE FROM scheduler\n               WHERE active = %s\n                 AND interval_ms = %s\n                 AND filename = %s'
        query_data = [self.active, self.interval_ms, self.filename]
        for col, val in iteritems(self.config_data):
            if val is not None:
                query_data.append(val)
                query_string += '\n  AND ' + col + ' = %s'
        cursor.execute(query_string, query_data)
        check_count = cursor.rowcount
        return (True, int(check_count))

    def manage_config(self, cursor, state):
        if state:
            if self.save_to_disk:
                save_config_to_disk(cursor, 'SCHEDULER')
            if self.load_to_runtime:
                load_config_to_runtime(cursor, 'SCHEDULER')

    def create_schedule(self, check_mode, result, cursor):
        if not check_mode:
            result['changed'] = self.create_schedule_config(cursor)
            result['msg'] = 'Added schedule to scheduler'
            result['schedules'] = self.get_schedule_config(cursor)
            self.manage_config(cursor, result['changed'])
        else:
            result['changed'] = True
            result['msg'] = 'Schedule would have been added to' + ' scheduler, however check_mode' + ' is enabled.'

    def delete_schedule(self, check_mode, result, cursor):
        if not check_mode:
            result['schedules'] = self.get_schedule_config(cursor)
            result['changed'] = self.delete_schedule_config(cursor)
            result['msg'] = 'Deleted schedule from scheduler'
            self.manage_config(cursor, result['changed'])
        else:
            result['changed'] = True
            result['msg'] = 'Schedule would have been deleted from' + ' scheduler, however check_mode is' + ' enabled.'