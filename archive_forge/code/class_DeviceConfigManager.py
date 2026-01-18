from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import (
from ansible.module_utils.six import iteritems
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
class DeviceConfigManager(BaseManager):
    """Manages AFM DoS Device Configuration settings.

    DeviceConfiguration is a special type of profile that is specific to the
    BIG-IP device's management interface; not the data plane interfaces.

    There are many similar vectors that can be managed here. This configuration
    is a super-set of the base DoS profile vector configuration and includes
    several attributes per-vector that are not found in the DoS profile configuration.
    These include,

      * allowUpstreamScrubbing
      * attackedDst
      * autoScrubbing
      * defaultInternalRateLimit
      * detectionThresholdPercent
      * detectionThresholdPps
      * perDstIpDetectionPps
      * perDstIpLimitPps
      * scrubbingDetectionSeconds
      * scrubbingDuration
    """

    def __init__(self, *args, **kwargs):
        super(DeviceConfigManager, self).__init__(**kwargs)
        self.want = ModuleParameters(params=self.module.params)
        self.have = ApiParameters()
        self.changes = UsableChanges()

    def update(self):
        name = self.normalize_names_in_device_config(self.want.name)
        self.want.update({'name': name})
        return self._update('dosDeviceVector')

    def normalize_names_in_device_config(self, name):
        name_map = {'hop-cnt-low': 'hop-cnt-leq-one', 'ip-low-ttl': 'ttl-leq-one'}
        result = name_map.get(name, name)
        return result

    def update_on_device(self):
        params = self.changes.api_params()
        uri = 'https://{0}:{1}/mgmt/tm/security/dos/device-config/{2}'.format(self.client.provider['server'], self.client.provider['server_port'], transform_name('Common', 'dos-device-config'))
        resp = self.client.api.patch(uri, json=params)
        try:
            response = resp.json()
        except ValueError as ex:
            raise F5ModuleError(str(ex))
        if resp.status in [200, 201] or ('code' in response and response['code'] in [200, 201]):
            return True
        raise F5ModuleError(resp.content)

    def read_current_from_device(self):
        uri = 'https://{0}:{1}/mgmt/tm/security/dos/device-config/{2}'.format(self.client.provider['server'], self.client.provider['server_port'], transform_name('Common', 'dos-device-config'))
        resp = self.client.api.get(uri)
        try:
            response = resp.json()
        except ValueError as ex:
            raise F5ModuleError(str(ex))
        if resp.status in [200, 201] or ('code' in response and response['code'] in [200, 201]):
            result = response.get('dosDeviceVector', [])
            return result
        raise F5ModuleError(resp.content)