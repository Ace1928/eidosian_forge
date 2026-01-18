from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.zabbix.plugins.module_utils.base import ZabbixBase
import ansible_collections.community.zabbix.plugins.module_utils.helpers as zabbix_utils
class Autoregistration(ZabbixBase):

    def get_autoregistration(self):
        try:
            return self._zapi.autoregistration.get({'output': 'extend'})
        except Exception as e:
            self._module.fail_json(msg='Failed to get autoregistration: %s' % e)

    def update_autoregistration(self, current_setting, tls_accept, tls_psk_identity, tls_psk):
        tls_accept_values = [None, 'unsecure', 'tls_with_psk']
        params = {}
        try:
            if isinstance(tls_accept, str):
                params['tls_accept'] = zabbix_utils.helper_to_numeric_value(tls_accept_values, tls_accept)
            elif isinstance(tls_accept, list):
                params['tls_accept'] = 0
                for _tls_accept_value in tls_accept:
                    params['tls_accept'] += zabbix_utils.helper_to_numeric_value(tls_accept_values, _tls_accept_value)
            else:
                self._module.fail_json(msg='Value of tls_accept must be list or string.')
            if tls_psk_identity:
                params['tls_psk_identity'] = tls_psk_identity
            if tls_psk:
                params['tls_psk'] = tls_psk
            current_tls_accept = int(current_setting['tls_accept'])
            if current_tls_accept == tls_accept_values.index('unsecure') and params['tls_accept'] >= tls_accept_values.index('tls_with_psk'):
                if not tls_psk_identity or not tls_psk:
                    self._module.fail_json(msg='Please set tls_psk_identity and tls_psk.')
            if not tls_psk_identity and (not tls_psk) and (params['tls_accept'] == current_tls_accept):
                self._module.exit_json(changed=False, result='Autoregistration is already up to date')
            if self._module.check_mode:
                self._module.exit_json(changed=True)
            self._zapi.autoregistration.update(params)
            self._module.exit_json(changed=True, result='Successfully updated global autoregistration setting')
        except Exception as e:
            self._module.fail_json(msg='Failed to update autoregistration: %s' % e)