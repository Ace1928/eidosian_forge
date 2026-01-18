from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule, env_fallback
from ansible_collections.community.digitalocean.plugins.module_utils.digital_ocean import (
class DOMonitoringAlertsInfo(object):

    def __init__(self, module):
        self.rest = DigitalOceanHelper(module)
        self.module = module
        self.module.params.pop('oauth_token')

    def get_alerts(self):
        alerts = self.rest.get_paginated_data(base_url='monitoring/alerts?', data_key_name='policies')
        self.module.exit_json(changed=False, data=alerts)

    def get_alert(self, uuid):
        alerts = self.rest.get_paginated_data(base_url='monitoring/alerts?', data_key_name='policies')
        for alert in alerts:
            alert_uuid = alert.get('uuid', None)
            if alert_uuid is not None:
                if alert_uuid == uuid:
                    self.module.exit_json(changed=False, data=alert)
            else:
                self.module.fail_json(changed=False, msg='Unexpected error; please file a bug: get_alert')
        self.module.exit_json(changed=False, data=[])