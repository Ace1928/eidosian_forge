from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
def __get_conn(self):
    self.conn = RHEVConn(self.module)
    return self.conn