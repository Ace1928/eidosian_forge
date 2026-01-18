from __future__ import absolute_import, division, print_function
from decimal import Decimal
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.mysql.plugins.module_utils.mysql import (
from ansible_collections.community.mysql.plugins.module_utils.user import (
from ansible.module_utils.six import iteritems
from ansible.module_utils._text import to_native
def __get_users_info(self):
    """Get user privileges, passwords, resources_limits, ...

        Query the server to get all the users and return a string
        of privileges that can be used by the mysql_user plugin.
        For instance:

        "users_info": [
            {
                "host": "users_info.com",
                "priv": "*.*: ALL,GRANT",
                "name": "users_info_adm"
            },
            {
                "host": "users_info.com",
                "priv": "`mysql`.*: SELECT/`users_info_db`.*: SELECT",
                "name": "users_info_multi"
            }
        ]
        """
    res = self.__exec_sql('SELECT * FROM mysql.user')
    if not res:
        return None
    output = list()
    for line in res:
        user = line['User']
        host = line['Host']
        user_priv = privileges_get(self.cursor, user, host)
        if not user_priv:
            self.module.warn('No privileges found for %s on host %s' % (user, host))
            continue
        priv_string = list()
        for db_table, priv in user_priv.items():
            if set(priv) == {'PROXY', 'GRANT'} or set(priv) == {'PROXY'}:
                continue
            unquote_db_table = db_table.replace('`', '').replace("'", '')
            priv_string.append('%s:%s' % (unquote_db_table, ','.join(priv)))
        if len(priv_string) > 1 and '*.*:USAGE' in priv_string:
            priv_string.remove('*.*:USAGE')
        resource_limits = get_resource_limits(self.cursor, user, host)
        copy_ressource_limits = dict.copy(resource_limits)
        output_dict = {'name': user, 'host': host, 'priv': '/'.join(priv_string), 'resource_limits': copy_ressource_limits}
        if resource_limits:
            for key, value in resource_limits.items():
                if value == 0:
                    del output_dict['resource_limits'][key]
            if len(output_dict['resource_limits']) == 0:
                del output_dict['resource_limits']
        authentications = get_existing_authentication(self.cursor, user, host)
        if authentications:
            output_dict.update(authentications)
        output.append(output_dict)
    self.info['users_info'] = output