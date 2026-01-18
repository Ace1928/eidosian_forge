from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.compare import compare_dictionary
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
def _transform_log_publisher(self, log_publisher):
    if log_publisher is None:
        return None
    if log_publisher in ['', 'none']:
        return {}
    return fq_name(self.partition, log_publisher)