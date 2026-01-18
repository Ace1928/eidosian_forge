from __future__ import (absolute_import, division, print_function)
import traceback
from ansible.errors import AnsibleActionFail, AnsibleConnectionFailure
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible.module_utils.common.validation import check_type_bool, check_type_int
from ansible.plugins.action import ActionBase
from ansible.utils.display import Display
from ..plugin_utils._reboot import reboot_host
def _process_download_install_result(self, task, result, inventory_hostname):
    phase = task[:-7]
    display.vv('Update phase %s completed' % phase, host=inventory_hostname)
    self._update_display_fired = 25
    total_results = self.download_results if phase == 'download' else self.install_results
    for result_info in result['info']:
        result_info['hresult'] = result_info['hresult'] & 4294967295
        total_results[result_info.pop('id')] = result_info