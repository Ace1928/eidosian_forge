from __future__ import absolute_import, division, print_function
from decimal import Decimal
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.mysql.plugins.module_utils.mysql import (
from ansible_collections.community.mysql.plugins.module_utils.user import (
from ansible.module_utils.six import iteritems
from ansible.module_utils._text import to_native
def __collect(self, exclude_fields, return_empty_dbs, wanted):
    """Collect all possible subsets."""
    if 'version' in wanted or 'settings' in wanted:
        self.__get_global_variables()
    if 'databases' in wanted:
        self.__get_databases(exclude_fields, return_empty_dbs)
    if 'global_status' in wanted:
        self.__get_global_status()
    if 'engines' in wanted:
        self.__get_engines()
    if 'users' in wanted:
        self.__get_users()
    if 'users_info' in wanted:
        self.__get_users_info()
    if 'master_status' in wanted:
        self.__get_master_status()
    if 'slave_status' in wanted:
        self.__get_slave_status()
    if 'slave_hosts' in wanted:
        self.__get_slaves()