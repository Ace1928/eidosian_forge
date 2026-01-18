from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.manageiq import ManageIQ, manageiq_argument_spec
class ManageIQAlerts(object):
    """ Object to execute alert management operations in manageiq.
    """

    def __init__(self, manageiq):
        self.manageiq = manageiq
        self.module = self.manageiq.module
        self.api_url = self.manageiq.api_url
        self.client = self.manageiq.client
        self.alerts_url = '{api_url}/alert_definitions'.format(api_url=self.api_url)

    def get_alerts(self):
        """ Get all alerts from ManageIQ
        """
        try:
            response = self.client.get(self.alerts_url + '?expand=resources')
        except Exception as e:
            self.module.fail_json(msg='Failed to query alerts: {error}'.format(error=e))
        return response.get('resources', [])

    def validate_hash_expression(self, expression):
        """ Validate a 'hash expression' alert definition
        """
        for key in ['options', 'eval_method', 'mode']:
            if key not in expression:
                msg = 'Hash expression is missing required field {key}'.format(key=key)
                self.module.fail_json(msg)

    def create_alert_dict(self, params):
        """ Create a dict representing an alert
        """
        if params['expression_type'] == 'hash':
            self.validate_hash_expression(params['expression'])
            expression_type = 'hash_expression'
        else:
            expression_type = 'expression'
        alert = dict(description=params['description'], db=params['resource_type'], options=params['options'], enabled=params['enabled'])
        alert.update({expression_type: params['expression']})
        return alert

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

    def delete_alert(self, alert):
        """ Delete an alert
        """
        try:
            result = self.client.post('{url}/{id}'.format(url=self.alerts_url, id=alert['id']), action='delete')
            msg = 'Alert {description} deleted: {details}'
            msg = msg.format(description=alert['description'], details=result)
            return dict(changed=True, msg=msg)
        except Exception as e:
            msg = 'Deleting alert {description} failed: {error}'
            msg = msg.format(description=alert['description'], error=e)
            self.module.fail_json(msg=msg)

    def update_alert(self, existing_alert, new_alert):
        """ Update an existing alert with the values from `new_alert`
        """
        new_alert_obj = ManageIQAlert(new_alert)
        if new_alert_obj == ManageIQAlert(existing_alert):
            return dict(changed=False, msg='No update needed')
        else:
            try:
                url = '{url}/{id}'.format(url=self.alerts_url, id=existing_alert['id'])
                result = self.client.post(url, action='edit', resource=new_alert)
                if new_alert_obj == ManageIQAlert(result):
                    msg = 'Alert {description} updated successfully: {details}'
                    msg = msg.format(description=existing_alert['description'], details=result)
                    return dict(changed=True, msg=msg)
                else:
                    msg = 'Updating alert {description} failed, unexpected result {details}'
                    msg = msg.format(description=existing_alert['description'], details=result)
                    self.module.fail_json(msg=msg)
            except Exception as e:
                msg = 'Updating alert {description} failed: {error}'
                if 'Resource expression needs be specified' in str(e):
                    msg = msg.format(description=existing_alert['description'], error='Your version of ManageIQ does not support hash_expression')
                else:
                    msg = msg.format(description=existing_alert['description'], error=e)
                self.module.fail_json(msg=msg)