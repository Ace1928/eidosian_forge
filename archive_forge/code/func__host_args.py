from __future__ import absolute_import, division, print_function
from datetime import datetime
import re
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import binary_type, text_type
def _host_args(self):
    """ Returns a normalized list of commandline arguments based on the "host" attribute """
    if self.host is None:
        return []
    elif self.host == 'currentHost':
        return ['-currentHost']
    else:
        return ['-host', self.host]