from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.compare import cmp_str_with_none
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
def _validate_creation_parameters(self):
    if self.want.protocol is None:
        raise F5ModuleError("'protocol' is required when creating a new ipfix destination.")
    if self.want.pool is None:
        raise F5ModuleError("'port' is required when creating a new ipfix destination.")
    if self.want.transport_profile is None:
        raise F5ModuleError("'transport_profile' is required when creating a new ipfix destination.")