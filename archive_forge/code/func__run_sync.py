from __future__ import (absolute_import, division, print_function)
import traceback
from ansible.errors import AnsibleActionFail, AnsibleConnectionFailure
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible.module_utils.common.validation import check_type_bool, check_type_int
from ansible.plugins.action import ActionBase
from ansible.utils.display import Display
from ..plugin_utils._reboot import reboot_host
def _run_sync(self, task_vars, module_options, reboot, reboot_timeout):
    """Installs the updates in a synchronous fashion with multiple update invocations if needed."""
    result = {'changed': False, 'reboot_required': False, 'rebooted': False}
    installed_updates = set()
    has_rebooted_on_failure = False
    round = 0
    while True:
        round += 1
        display.v('Running win_updates - round %s' % round, host=task_vars.get('inventory_hostname', None))
        update_result = self._run_updates(task_vars, module_options)
        self._updates.update(update_result.updates)
        self._filtered_updates.update(update_result.filtered_updates)
        self._selected_updates.update(update_result.selected_updates)
        self._download_results.update(update_result.download_results)
        self._install_results.update(update_result.install_results)
        current_updates = set(update_result.install_results)
        new_updates = current_updates.difference(installed_updates)
        installed_updates.update(current_updates)
        if current_updates and (not new_updates):
            for update_id in current_updates:
                self._install_results[update_id]['result_code'] = 4
                self._install_results[update_id]['hresult'] = -1
            result['failed'] = True
            result['msg'] = f'An update loop was detected, this could be caused by an update being rolled back during a reboot or the Windows Update API incorrectly reporting a failed update as being successful.Check the Windows Updates logs on the host to gather more information. Updates in the reboot loop are: {', '.join(current_updates)}'
            break
        reboot_required = result['reboot_required'] = update_result.reboot_required
        if update_result.changed:
            result['changed'] = True
        if update_result.failed:
            msg = update_result.msg
            if update_result.hresult:
                msg += ' - ' + _get_hresult_error(update_result.hresult)
            if reboot and (not has_rebooted_on_failure) and (module_options.get('state', '') != 'searched'):
                display.vv('Failure when running win_updates module (Will retry after reboot): %s' % msg, host=task_vars.get('inventory_hostname', None))
                reboot_required = True
                has_rebooted_on_failure = True
            else:
                result['failed'] = True
                result['msg'] = msg
                result['exception'] = update_result.exception
                break
        elif reboot:
            has_rebooted_on_failure = False
        if reboot_required and reboot:
            display.v('Rebooting host after installing updates', host=task_vars.get('inventory_hostname', None))
            if self._play_context.check_mode:
                reboot_res = {'failed': False}
            else:
                reboot_res = reboot_host(self._task.action, self._connection, reboot_timeout=reboot_timeout)
            result['rebooted'] = True
            if reboot_res['failed']:
                msg = 'Failed to reboot host'
                if 'msg' in reboot_res:
                    msg += ': ' + str(reboot_res['msg'])
                reboot_res['msg'] = msg
                result.update(reboot_res)
                break
            result['changed'] = True
            result['reboot_required'] = False
        if not reboot or self._play_context.check_mode or module_options.get('state', 'installed') != 'installed' or (not reboot_required and len(update_result.selected_updates) == 0):
            break
    return result