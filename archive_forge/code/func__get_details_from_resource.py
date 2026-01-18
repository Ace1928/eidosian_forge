from __future__ import absolute_import, division, print_function
import re
import time
from datetime import datetime
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
def _get_details_from_resource(self):
    resource = self.read_current_from_device()
    stats = resource['entries'].copy()
    if HAS_OBJPATH:
        tree = Tree(stats)
    else:
        raise F5ModuleError('objectpath module required, install objectpath module to continue. ')
    details = list(tree.execute('$..*["details"]["description"]'))
    result = details[::-1]
    return result