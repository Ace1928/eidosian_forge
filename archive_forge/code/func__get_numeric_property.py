from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.compare import cmp_str_with_none
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
def _get_numeric_property(self, property):
    if self._values[property] is None:
        return None
    try:
        fvar = float(self._values[property])
    except ValueError:
        raise F5ModuleError('Provided {0} must be a valid number'.format(property))
    return fvar