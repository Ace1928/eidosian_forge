from __future__ import (absolute_import, division, print_function)
import uuid
from ansible.module_utils.basic import AnsibleModule, env_fallback
from ansible.module_utils.common.text.converters import to_native
def act_on_volume(target_state, module, packet_conn):
    return_dict = {'changed': False}
    s = get_volume_selector(module)
    project_id = module.params.get('project_id')
    api_method = 'projects/{0}/storage'.format(project_id)
    all_volumes = packet_conn.call_api(api_method, 'GET')['volumes']
    matching_volumes = [v for v in all_volumes if s(v)]
    if target_state == 'present':
        if len(matching_volumes) == 0:
            params = {'description': get_or_fail(module.params, 'description'), 'size': get_or_fail(module.params, 'size'), 'plan': get_or_fail(module.params, 'plan'), 'facility': get_or_fail(module.params, 'facility'), 'locked': get_or_fail(module.params, 'locked'), 'billing_cycle': get_or_fail(module.params, 'billing_cycle'), 'snapshot_policies': module.params.get('snapshot_policy')}
            new_volume_data = packet_conn.call_api(api_method, 'POST', params)
            return_dict['changed'] = True
            for k in ['id', 'name', 'description']:
                return_dict[k] = new_volume_data[k]
        else:
            for k in ['id', 'name', 'description']:
                return_dict[k] = matching_volumes[0][k]
    else:
        if len(matching_volumes) > 1:
            _msg = 'More than one volume matches in module call for absent state: {0}'.format(to_native(matching_volumes))
            module.fail_json(msg=_msg)
        if len(matching_volumes) == 1:
            volume = matching_volumes[0]
            packet_conn.call_api('storage/{0}'.format(volume['id']), 'DELETE')
            return_dict['changed'] = True
            for k in ['id', 'name', 'description']:
                return_dict[k] = volume[k]
    return return_dict