from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
def _update_downtime(module, current_downtime, api_client):
    api = DowntimesApi(api_client)
    downtime = build_downtime(module)
    try:
        if current_downtime.disabled:
            resp = api.create_downtime(downtime)
        else:
            resp = api.update_downtime(module.params['id'], downtime)
        if _equal_dicts(resp.to_dict(), current_downtime.to_dict(), ['active', 'creator_id', 'updater_id']):
            module.exit_json(changed=False, downtime=resp.to_dict())
        else:
            module.exit_json(changed=True, downtime=resp.to_dict())
    except ApiException as e:
        module.fail_json(msg='Failed to update downtime: {0}'.format(e))