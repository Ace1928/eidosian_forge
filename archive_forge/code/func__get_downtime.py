from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
def _get_downtime(module, api_client):
    api = DowntimesApi(api_client)
    downtime = None
    if module.params['id']:
        try:
            downtime = api.get_downtime(module.params['id'])
        except ApiException as e:
            module.fail_json(msg='Failed to retrieve downtime with id {0}: {1}'.format(module.params['id'], e))
    return downtime