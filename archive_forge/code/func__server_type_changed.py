from __future__ import absolute_import, division, print_function
import re
import traceback
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import (
from ..module_utils.teem import send_teem
def _server_type_changed(self):
    if self.want.server_type is None:
        self.want.update({'server_type': self.have.server_type})
    if self.want.server_type != self.have.server_type:
        return True
    return False