from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.basic import env_fallback
class VdirectCommit(object):

    def __init__(self, params):
        self.client = rest_client.RestClient(params['vdirect_ip'], params['vdirect_user'], params['vdirect_password'], wait=params['vdirect_wait'], secondary_vdirect_ip=params['vdirect_secondary_ip'], https_port=params['vdirect_https_port'], http_port=params['vdirect_http_port'], timeout=params['vdirect_timeout'], https=params['vdirect_use_ssl'], verify=params['validate_certs'])
        self.devices = params['devices']
        self.apply = params['apply']
        self.save = params['save']
        self.sync = params['sync']
        self.devicesMap = {}

    def _validate_devices(self):
        for device in self.devices:
            try:
                res = self.client.adc.get(device)
                if res[rest_client.RESP_STATUS] == 200:
                    self.devicesMap.update({device: ADC_DEVICE_TYPE})
                    continue
                res = self.client.container.get(device)
                if res[rest_client.RESP_STATUS] == 200:
                    if res[rest_client.RESP_DATA]['type'] == PARTITIONED_CONTAINER_DEVICE_TYPE:
                        self.devicesMap.update({device: CONTAINER_DEVICE_TYPE})
                    continue
                res = self.client.appWall.get(device)
                if res[rest_client.RESP_STATUS] == 200:
                    self.devicesMap.update({device: APPWALL_DEVICE_TYPE})
                    continue
                res = self.client.defensePro.get(device)
                if res[rest_client.RESP_STATUS] == 200:
                    self.devicesMap.update({device: DP_DEVICE_TYPE})
                    continue
            except Exception as e:
                raise CommitException('Failed to communicate with device ' + device, str(e))
            raise MissingDeviceException(device)

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

    def commit(self):
        self._validate_devices()
        result_to_return = dict()
        result_to_return['details'] = list()
        for device in self.devices:
            failure_occurred = False
            device_type = self.devicesMap[device]
            actions_result = dict()
            actions_result['device_name'] = device
            actions_result['device_type'] = device_type
            if device_type in [DP_DEVICE_TYPE, APPWALL_DEVICE_TYPE]:
                failure_occurred = not self._perform_action_and_update_result(device, 'commit', True, failure_occurred, actions_result) or failure_occurred
            else:
                failure_occurred = not self._perform_action_and_update_result(device, 'apply', self.apply, failure_occurred, actions_result) or failure_occurred
                if device_type != CONTAINER_DEVICE_TYPE:
                    failure_occurred = not self._perform_action_and_update_result(device, 'sync', self.sync, failure_occurred, actions_result) or failure_occurred
                failure_occurred = not self._perform_action_and_update_result(device, 'save', self.save, failure_occurred, actions_result) or failure_occurred
            result_to_return['details'].extend([actions_result])
            if failure_occurred:
                result_to_return['msg'] = FAILURE
        if 'msg' not in result_to_return:
            result_to_return['msg'] = SUCCESS
        return result_to_return