from __future__ import absolute_import, division, print_function
import os
import uuid
from ansible.module_utils.basic import AnsibleModule
def act_on_sshkeys(target_state, module, packet_conn):
    selector = get_sshkey_selector(module)
    existing_sshkeys = packet_conn.list_ssh_keys()
    matching_sshkeys = filter(selector, existing_sshkeys)
    changed = False
    if target_state == 'present':
        if matching_sshkeys == []:
            newkey = {}
            if module.params.get('key_file'):
                with open(module.params.get('key_file')) as f:
                    newkey = load_key_string(f.read())
            if module.params.get('key'):
                newkey = load_key_string(module.params.get('key'))
            if module.params.get('label'):
                newkey['label'] = module.params.get('label')
            for param in ('label', 'key'):
                if param not in newkey:
                    _msg = 'If you want to ensure a key is present, you must supply both a label and a key string, either in module params, or in a key file. %s is missing' % param
                    raise Exception(_msg)
            matching_sshkeys = []
            new_key_response = packet_conn.create_ssh_key(newkey['label'], newkey['key'])
            changed = True
            matching_sshkeys.append(new_key_response)
    else:
        for k in matching_sshkeys:
            try:
                k.delete()
                changed = True
            except Exception as e:
                _msg = 'while trying to remove sshkey %s, id %s %s, got error: %s' % (k.label, k.id, target_state, e)
                raise Exception(_msg)
    return {'changed': changed, 'sshkeys': [serialize_sshkey(k) for k in matching_sshkeys]}