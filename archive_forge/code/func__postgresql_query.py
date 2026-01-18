from __future__ import absolute_import, division, print_function
import types
from ansible.errors import AnsibleFilterError
from ansible.module_utils.basic import missing_required_lib
from ansible.utils.display import Display
def _postgresql_query(v):
    v.dialect = netaddr.mac_pgsql
    return str(v)