from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
def absent_folder(module):
    if module.params['grafana_url'][-1] == '/':
        module.params['grafana_url'] = module.params['grafana_url'][:-1]
    sameConfig = False
    api_url = module.params['grafana_url'] + '/api/folders'
    result = requests.get(api_url, headers={'Authorization': 'Bearer ' + module.params['grafana_api_key']})
    for folder in result.json():
        if folder['uid'] == module.params['uid'] and folder['title'] == module.params['title']:
            sameConfig = True
    if sameConfig is True:
        api_url = module.params['grafana_url'] + '/api/folders/' + module.params['uid']
        result = requests.delete(api_url, headers={'Authorization': 'Bearer ' + module.params['grafana_api_key']})
        if result.status_code == 200:
            return (False, True, {'status': result.status_code, 'response': 'Folder has been succesfuly deleted'})
        else:
            return (True, False, {'status': result.status_code, 'response': 'Error deleting folder'})
    else:
        return (False, True, {'status': 200, 'response': 'Folder does not exist'})