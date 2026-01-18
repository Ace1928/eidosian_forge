from __future__ import absolute_import, division, print_function
import uuid
import re
from ansible.module_utils.basic import AnsibleModule, env_fallback
from ansible.module_utils.common.text.converters import to_native
def act_on_assignment(target_state, module, packet_conn):
    return_dict = {'changed': False}
    specified_cidr = module.params.get('cidr')
    address, prefixlen = parse_subnet_cidr(specified_cidr)
    specified_identifier = get_specified_device_identifiers(module)
    if module.check_mode:
        return return_dict
    if specified_identifier['hostname'] is None and specified_identifier['device_id'] is None:
        if target_state == 'absent':
            for d in get_existing_devices(module, packet_conn):
                for ia in d.ip_addresses:
                    if address == ia['address'] and prefixlen == ia['cidr']:
                        packet_conn.call_api(ia['href'], 'DELETE')
                        return_dict['changed'] = True
                        return_dict['subnet'] = ia
                        return_dict['device_id'] = d.id
                        return return_dict
        raise Exception('If you assign an address, you must specify either target device ID or target unique hostname.')
    if specified_identifier['device_id'] is not None:
        device = packet_conn.get_device(specified_identifier['device_id'])
    else:
        all_devices = get_existing_devices(module, packet_conn)
        hn = specified_identifier['hostname']
        matching_devices = [d for d in all_devices if d.hostname == hn]
        if len(matching_devices) > 1:
            raise Exception('There are more than one devices matching given hostname {0}'.format(hn))
        if len(matching_devices) == 0:
            raise Exception('There is no device matching given hostname {0}'.format(hn))
        device = matching_devices[0]
    return_dict['device_id'] = device.id
    assignment_dicts = [i for i in device.ip_addresses if i['address'] == address and i['cidr'] == prefixlen]
    if len(assignment_dicts) > 1:
        raise Exception('IP address {0} is assigned more than once for device {1}'.format(specified_cidr, device.hostname))
    if target_state == 'absent':
        if len(assignment_dicts) == 1:
            packet_conn.call_api(assignment_dicts[0]['href'], 'DELETE')
            return_dict['subnet'] = assignment_dicts[0]
            return_dict['changed'] = True
    elif target_state == 'present':
        if len(assignment_dicts) == 0:
            new_assignment = packet_conn.call_api('devices/{0}/ips'.format(device.id), 'POST', {'address': '{0}'.format(specified_cidr)})
            return_dict['changed'] = True
            return_dict['subnet'] = new_assignment
    return return_dict