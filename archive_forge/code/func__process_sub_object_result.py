from __future__ import (absolute_import, division, print_function)
import os
import time
import traceback
from ansible.module_utils._text import to_text
import json
from ansible_collections.fortinet.fortios.plugins.module_utils.common.type_utils import underscore_to_hyphen
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.secret_field import is_secret_field
def _process_sub_object_result(self, results):
    meta = list()
    failed = False
    changed = False
    for result in results:
        sub_obj = result[0]
        result_data = result[1]
        url = sub_obj['get']
        suffix_index = url.find('?')
        if suffix_index >= 0:
            url = url[:suffix_index]
        result_data['object_path'] = url[12:]
        meta.append(result_data)
        if 'status' in result_data:
            if result_data['status'] == 'error':
                failed = True
            elif result_data['status'] == 'success':
                if 'revision_changed' in result_data and result_data['revision_changed'] is True:
                    changed = True
                elif 'revision_changed' not in result_data:
                    changed = True
    self._module.exit_json(meta=meta, changed=changed, failed=failed)