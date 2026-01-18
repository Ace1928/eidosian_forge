from __future__ import (absolute_import, division, print_function)
import traceback
from ansible.errors import AnsibleActionFail, AnsibleConnectionFailure
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible.module_utils.common.validation import check_type_bool, check_type_int
from ansible.plugins.action import ActionBase
from ansible.utils.display import Display
from ..plugin_utils._reboot import reboot_host
def _process_download_install_progress(self, task, result, inventory_hostname):
    update_id = result['progress']['CurrentUpdateId']
    update = self.updates[update_id]
    total_percentage = result['progress']['PercentComplete']
    if task == 'download':
        current_phase = result['progress']['CurrentUpdateDownloadPhase']
        download_phase = {1: 'Initializing', 2: 'Downloading', 3: 'Verifying'}.get(current_phase, current_phase)
        msg = 'Download progress - Total: {0}/{1} {2}%, Update ({3}): {4}/{5} {6}%, Phase: {7}'.format(result['progress']['TotalBytesDownloaded'], result['progress']['TotalBytesToDownload'], total_percentage, update['title'], result['progress']['CurrentUpdateBytesDownloaded'], result['progress']['CurrentUpdateBytesToDownload'], result['progress']['CurrentUpdatePercentComplete'], download_phase)
    else:
        msg = 'Install progress - Total: {0}%, Update ({1}): {2}%'.format(total_percentage, update['title'], result['progress']['CurrentUpdatePercentComplete'])
    if total_percentage >= self._update_display_fired:
        display.vv(msg, host=inventory_hostname)
        while self._update_display_fired <= total_percentage:
            self._update_display_fired += 25
    display.debug(msg, host=inventory_hostname)