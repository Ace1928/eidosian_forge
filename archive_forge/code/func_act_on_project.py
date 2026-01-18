from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule, env_fallback
from ansible.module_utils.common.text.converters import to_native
def act_on_project(target_state, module, packet_conn):
    result_dict = {'changed': False}
    given_id = module.params.get('id')
    given_name = module.params.get('name')
    if given_id:
        matching_projects = [p for p in packet_conn.list_projects() if given_id == p.id]
    else:
        matching_projects = [p for p in packet_conn.list_projects() if given_name == p.name]
    if target_state == 'present':
        if len(matching_projects) == 0:
            org_id = module.params.get('org_id')
            custom_data = module.params.get('custom_data')
            payment_method = module.params.get('payment_method')
            if not org_id:
                params = {'name': given_name, 'payment_method_id': payment_method, 'customdata': custom_data}
                new_project_data = packet_conn.call_api('projects', 'POST', params)
                new_project = packet.Project(new_project_data, packet_conn)
            else:
                new_project = packet_conn.create_organization_project(org_id=org_id, name=given_name, payment_method_id=payment_method, customdata=custom_data)
            result_dict['changed'] = True
            matching_projects.append(new_project)
        result_dict['name'] = matching_projects[0].name
        result_dict['id'] = matching_projects[0].id
    else:
        if len(matching_projects) > 1:
            _msg = 'More than projects matched for module call with state = absent: {0}'.format(to_native(matching_projects))
            module.fail_json(msg=_msg)
        if len(matching_projects) == 1:
            p = matching_projects[0]
            result_dict['name'] = p.name
            result_dict['id'] = p.id
            result_dict['changed'] = True
            try:
                p.delete()
            except Exception as e:
                _msg = 'while trying to remove project {0}, id {1}, got error: {2}'.format(p.name, p.id, to_native(e))
                module.fail_json(msg=_msg)
    return result_dict