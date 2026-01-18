from __future__ import absolute_import, division, print_function
import re
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.resource_module_base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
def _deepformat(self, tmplt, data):
    wtmplt = deepcopy(tmplt)
    if isinstance(tmplt, str):
        res = self._template(value=tmplt, variables=data, fail_on_undefined=False)
        return res
    if isinstance(tmplt, dict):
        for tkey, tval in tmplt.items():
            ftkey = self._template(tkey, data)
            if ftkey != tkey:
                wtmplt.pop(tkey)
            if isinstance(tval, dict):
                wtmplt[ftkey] = self._deepformat(tval, data)
            elif isinstance(tval, list):
                wtmplt[ftkey] = [self._deepformat(x, data) for x in tval]
            elif isinstance(tval, str):
                wtmplt[ftkey] = self._deepformat(tval, data)
                if wtmplt[ftkey] is None:
                    wtmplt.pop(ftkey)
    return wtmplt