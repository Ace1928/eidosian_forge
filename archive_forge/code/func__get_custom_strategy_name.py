from __future__ import absolute_import, division, print_function
import re
import traceback
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
def _get_custom_strategy_name(self):
    strategy = self._values['strategy']
    if re.match('(\\/[a-zA-Z_0-9.-]+){2}', strategy):
        return strategy
    elif re.match('[a-zA-Z_0-9.-]+', strategy):
        return '/{0}/{1}'.format(self.partition, strategy)
    else:
        raise F5ModuleError('The provided strategy name is invalid!')