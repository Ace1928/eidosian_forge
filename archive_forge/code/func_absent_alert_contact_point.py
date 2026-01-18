from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
def absent_alert_contact_point(module):
    already_exists = False
    if module.params['grafana_url'][-1] == '/':
        module.params['grafana_url'] = module.params['grafana_url'][:-1]
    api_url = module.params['grafana_url'] + '/api/v1/provisioning/contact-points'
    result = requests.get(api_url, headers={'Authorization': 'Bearer ' + module.params['grafana_api_key']})
    for contact_points in result.json():
        if contact_points['uid'] == module.params['uid']:
            already_exists = True
    if already_exists:
        api_url = module.params['grafana_url'] + '/api/v1/provisioning/contact-points/' + module.params['uid']
        result = requests.delete(api_url, headers={'Authorization': 'Bearer ' + module.params['grafana_api_key']})
        if result.status_code == 202:
            return (False, True, result.json())
        else:
            return (True, False, {'status': result.status_code, 'response': result.json()['message']})
    else:
        return (True, False, 'Alert Contact point does not exist')