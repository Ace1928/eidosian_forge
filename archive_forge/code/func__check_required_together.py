from __future__ import absolute_import, division, print_function
import traceback
import re
import json
from itertools import chain
from ansible.module_utils.common.text.converters import to_text
from ansible.module_utils._text import to_native
from ansible.module_utils.common.collections import is_iterable
from ansible.module_utils.basic import AnsibleModule, missing_required_lib, _load_params
from ansible.module_utils.urls import open_url
def _check_required_together(self, spec, param=None):
    if spec is None:
        return
    if param is None:
        param = self.params
    try:
        self.check_required_together(spec, param)
    except TypeError as e:
        msg = to_native(e)
        if self._options_context:
            msg += ' found in %s' % ' -> '.join(self._options_context)
        self.fail_json(msg=msg)