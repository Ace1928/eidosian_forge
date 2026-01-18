from __future__ import absolute_import, division, print_function
import re
import uuid
from datetime import datetime
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.basic import env_fallback
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import (
from ..module_utils.teem import send_teem
def _get_temporary_template(self):
    self.create_on_device()
    temp = self.read_current_from_device()
    self.remove_from_device()
    return temp