from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.compare import cmp_str_with_none
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
def _internal_name(self):
    name = self.want.profile_name
    partition = self.want.partition
    if 'global-network' in name:
        return 'global-network'
    return transform_name(partition, name)