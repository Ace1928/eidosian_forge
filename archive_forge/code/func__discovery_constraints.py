from __future__ import absolute_import, division, print_function
import re
import traceback
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import (
from ..module_utils.teem import send_teem
def _discovery_constraints(self):
    if self.want.virtual_server_discovery is None:
        virtual_server_discovery = self.have.virtual_server_discovery
    else:
        virtual_server_discovery = self.want.virtual_server_discovery
    if self.want.link_discovery is None:
        link_discovery = self.have.link_discovery
    else:
        link_discovery = self.want.link_discovery
    if link_discovery in ['enabled', 'enabled-no-delete'] and virtual_server_discovery == 'disabled':
        raise F5ModuleError('Virtual server discovery must be enabled if link discovery is enabled')