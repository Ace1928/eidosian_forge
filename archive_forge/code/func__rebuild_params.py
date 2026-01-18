from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.compare import cmp_str_with_none
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
def _rebuild_params(self, enabled, rate_limit):
    to_filter = dict(enabled=flatten_boolean(self._values[enabled]), rate_limit=self._change_aggregate_rate_value(self._values[rate_limit]))
    result = self._filter_params(to_filter)
    return result