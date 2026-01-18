from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import _load_params
import datetime
from ansible.module_utils.six import raise_from
def _do_final_exit(self, rc, result, changed=True):
    failed = rc != 0
    if changed:
        changed = rc == 0
    if 'response_code' in result:
        if 'rc_failed' in self.module.params and self.module.params['rc_failed']:
            for rc_code in self.module.params['rc_failed']:
                if str(result['response_code']) == str(rc_code):
                    failed = True
                    result['result_code_overriding'] = 'rc code:%s is overridden to failure' % rc_code
        elif 'rc_succeeded' in self.module.params and self.module.params['rc_succeeded']:
            for rc_code in self.module.params['rc_succeeded']:
                if str(result['response_code']) == str(rc_code):
                    failed = False
                    result['result_code_overriding'] = 'rc code:%s is overridden to success' % rc_code
    if self.system_status:
        result['system_information'] = self.system_status
    if len(self.version_check_warnings):
        version_check_warning = dict()
        version_check_warning['mismatches'] = self.version_check_warnings
        if not self.system_status:
            raise AssertionError("Can't get system status, please check Internet connection.")
        version_check_warning['system_version'] = 'v%s.%s.%s' % (self.system_status['Major'], self.system_status['Minor'], self.system_status['Patch'])
        self.module.warn('Some parameters in the playbook may not be supported by the current FortiManager version. To see which parameters are not available, check version_check_warning in the output. This message is only a suggestion. Please ignore this warning if you think your cofigurations are correct.')
        self.module.exit_json(rc=rc, meta=result, version_check_warning=version_check_warning, failed=failed, changed=changed)
    else:
        self.module.exit_json(rc=rc, meta=result, failed=failed, changed=changed)