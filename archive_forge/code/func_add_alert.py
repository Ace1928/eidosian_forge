from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.manageiq import ManageIQ, manageiq_argument_spec
def add_alert(self, alert):
    """ Add a new alert to ManageIQ
        """
    try:
        result = self.client.post(self.alerts_url, action='create', resource=alert)
        msg = 'Alert {description} created successfully: {details}'
        msg = msg.format(description=alert['description'], details=result)
        return dict(changed=True, msg=msg)
    except Exception as e:
        msg = 'Creating alert {description} failed: {error}'
        if 'Resource expression needs be specified' in str(e):
            msg = msg.format(description=alert['description'], error='Your version of ManageIQ does not support hash_expression')
        else:
            msg = msg.format(description=alert['description'], error=e)
        self.module.fail_json(msg=msg)