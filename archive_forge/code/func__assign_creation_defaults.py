from __future__ import absolute_import, division, print_function
import re
import traceback
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import (
from ..module_utils.teem import send_teem
def _assign_creation_defaults(self):
    if self.want.server_type is None:
        self.want.update({'server_type': 'bigip'})
    if self.want.link_discovery is None:
        self.want.update({'link_discovery': 'disabled'})
    if self.want.virtual_server_discovery is None:
        self.want.update({'virtual_server_discovery': 'disabled'})
    self._check_link_discovery_requirements()