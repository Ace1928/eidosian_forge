from __future__ import absolute_import, division, print_function
from datetime import datetime
import re
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import binary_type, text_type
def _base_command(self):
    """ Returns a list containing the "defaults" executable and any common base arguments """
    return [self.executable] + self._host_args()