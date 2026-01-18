from __future__ import (absolute_import, division, print_function)
import traceback
from ansible.errors import AnsibleActionFail, AnsibleConnectionFailure
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible.module_utils.common.validation import check_type_bool, check_type_int
from ansible.plugins.action import ActionBase
from ansible.utils.display import Display
from ..plugin_utils._reboot import reboot_host
def _get_update_info(self, update_id):
    """Gets the update results info value to return."""
    raw_info = self._updates[update_id]
    info = {'title': raw_info['title'], 'kb': [v[2:] if v.startswith('KB') else v for v in raw_info['kb']], 'categories': raw_info['categories'], 'id': update_id, 'downloaded': False, 'installed': False}
    for action, results in [('downloaded', self._download_results), ('installed', self._install_results)]:
        action_info = results.get(update_id, None)
        if action_info:
            if action_info['result_code'] == 2:
                info[action] = True
            else:
                info['failure_hresult_code'] = action_info['hresult']
                info['failure_msg'] = _get_hresult_error(action_info['hresult'])
    return info