from __future__ import absolute_import, division, print_function
from ansible_collections.community.general.plugins.module_utils.scaleway import SCALEWAY_LOCATION, scaleway_argument_spec, Scaleway
from ansible.module_utils.basic import AnsibleModule
from uuid import uuid4
def absent_strategy(api, security_group):
    response = api.get('security_groups')
    ret = {'changed': False}
    if not response.ok:
        api.module.fail_json(msg='Error getting security groups "%s": "%s" (%s)' % (response.info['msg'], response.json['message'], response.json))
    security_group_lookup = dict(((sg['name'], sg) for sg in response.json['security_groups']))
    if security_group['name'] not in security_group_lookup.keys():
        return ret
    ret['changed'] = True
    if api.module.check_mode:
        return ret
    response = api.delete('/security_groups/' + security_group_lookup[security_group['name']]['id'])
    if not response.ok:
        api.module.fail_json(msg='Error deleting security group "%s": "%s" (%s)' % (response.info['msg'], response.json['message'], response.json))
    return ret