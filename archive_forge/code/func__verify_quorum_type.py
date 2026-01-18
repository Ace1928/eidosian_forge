from __future__ import absolute_import, division, print_function
import re
from copy import deepcopy
from datetime import datetime
from ansible.module_utils.urls import urlparse
from ansible.module_utils.basic import (
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import remove_default_spec
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.compare import cmp_str_with_none
from ..module_utils.icontrol import (
from ..module_utils.teem import send_teem
def _verify_quorum_type(self, quorum):
    try:
        if quorum is None:
            return None
        return int(quorum)
    except ValueError:
        raise F5ModuleError("The specified 'quorum' must be an integer.")