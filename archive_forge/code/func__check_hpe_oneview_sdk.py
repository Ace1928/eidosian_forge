from __future__ import (absolute_import, division, print_function)
import abc
import collections
import json
import os  # noqa: F401, pylint: disable=unused-import
import traceback
from ansible.module_utils import six
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.common._collections_compat import Mapping
def _check_hpe_oneview_sdk(self):
    if not HAS_HPE_ONEVIEW:
        self.module.fail_json(msg=missing_required_lib('hpOneView'), exception=HPE_ONEVIEW_IMP_ERR)