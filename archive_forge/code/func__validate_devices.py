from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.basic import env_fallback
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