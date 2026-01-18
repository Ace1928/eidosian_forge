from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.basic import env_fallback
def _perform_action_and_update_result(self, device, action, perform, failure_occurred, actions_result):
    if not perform or failure_occurred:
        actions_result[action] = NOT_PERFORMED
        return True
    try:
        if self.devicesMap[device] == ADC_DEVICE_TYPE:
            res = self.client.adc.control_device(device, action)
        elif self.devicesMap[device] == CONTAINER_DEVICE_TYPE:
            res = self.client.container.control(device, action)
        elif self.devicesMap[device] == APPWALL_DEVICE_TYPE:
            res = self.client.appWall.control_device(device, action)
        elif self.devicesMap[device] == DP_DEVICE_TYPE:
            res = self.client.defensePro.control_device(device, action)
        if res[rest_client.RESP_STATUS] in [200, 204]:
            actions_result[action] = SUCCEEDED
        else:
            actions_result[action] = FAILED
            actions_result['failure_description'] = res[rest_client.RESP_STR]
            return False
    except Exception as e:
        actions_result[action] = FAILED
        actions_result['failure_description'] = 'Exception occurred while performing ' + action + ' action. Exception: ' + str(e)
        return False
    return True