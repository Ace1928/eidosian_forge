from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
def absent_cloud_api_key(module):
    api_url = 'https://grafana.com/api/orgs/' + module.params['org_slug'] + '/api-keys/' + module.params['name']
    result = requests.delete(api_url, headers={'Authorization': 'Bearer ' + module.params['existing_cloud_api_key']})
    if result.status_code == 200:
        return (False, True, 'Cloud API key is deleted')
    else:
        return (True, False, {'status': result.status_code, 'response': result.json()['message']})