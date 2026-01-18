from __future__ import absolute_import, division, print_function
from datetime import datetime, timedelta
from time import sleep
from copy import deepcopy
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.urls import fetch_url
from ansible.module_utils._text import to_text
def _param_updated(self, key, resource):
    param = self._module.params.get(key)
    if param is None:
        return False
    if not resource or key not in resource:
        return False
    is_different = self.find_difference(key, resource, param)
    if is_different:
        self._result['changed'] = True
        patch_data = {key: param}
        self._result['diff']['before'].update({key: resource[key]})
        self._result['diff']['after'].update(patch_data)
        if not self._module.check_mode:
            href = resource.get('href')
            if not href:
                self._module.fail_json(msg='Unable to update %s, no href found.' % key)
            self._patch(href, patch_data)
            return True
    return False