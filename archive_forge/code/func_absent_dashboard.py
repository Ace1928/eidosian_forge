from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
def absent_dashboard(module):
    if module.params['grafana_url'][-1] == '/':
        module.params['grafana_url'] = module.params['grafana_url'][:-1]
    if 'uid' not in module.params['dashboard']['dashboard']:
        return (True, False, 'UID is not defined in the the Dashboard configuration')
    api_url = api_url = module.params['grafana_url'] + '/api/dashboards/uid/' + module.params['dashboard']['dashboard']['uid']
    result = requests.delete(api_url, headers={'Authorization': 'Bearer ' + module.params['grafana_api_key']})
    if result.status_code == 200:
        return (False, True, result.json())
    else:
        return (True, False, {'status': result.status_code, 'response': result.json()['message']})