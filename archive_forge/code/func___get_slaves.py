from __future__ import absolute_import, division, print_function
from decimal import Decimal
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.mysql.plugins.module_utils.mysql import (
from ansible_collections.community.mysql.plugins.module_utils.user import (
from ansible.module_utils.six import iteritems
from ansible.module_utils._text import to_native
def __get_slaves(self):
    """Get slave hosts info if the instance is a master."""
    res = self.__exec_sql('SHOW SLAVE HOSTS')
    if res:
        for line in res:
            srv_id = line['Server_id']
            if srv_id not in self.info['slave_hosts']:
                self.info['slave_hosts'][srv_id] = {}
            for vname, val in iteritems(line):
                if vname != 'Server_id':
                    self.info['slave_hosts'][srv_id][vname] = self.__convert(val)