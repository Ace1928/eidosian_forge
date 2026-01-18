from __future__ import absolute_import, division, print_function
import json
from ansible.module_utils.api import basic_auth_argument_spec
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.module_utils.urls import open_url
class AMGsync(object):

    def __init__(self):
        argument_spec = basic_auth_argument_spec()
        argument_spec.update(dict(api_username=dict(type='str', required=True), api_password=dict(type='str', required=True, no_log=True), api_url=dict(type='str', required=True), name=dict(required=True, type='str'), ssid=dict(required=True, type='str'), state=dict(required=True, type='str', choices=['running', 'suspended']), delete_recovery_point=dict(required=False, type='bool', default=False)))
        self.module = AnsibleModule(argument_spec=argument_spec)
        args = self.module.params
        self.name = args['name']
        self.ssid = args['ssid']
        self.state = args['state']
        self.delete_recovery_point = args['delete_recovery_point']
        try:
            self.user = args['api_username']
            self.pwd = args['api_password']
            self.url = args['api_url']
        except KeyError:
            self.module.fail_json(msg='You must pass in api_usernameand api_password and api_url to the module.')
        self.certs = args['validate_certs']
        self.post_headers = {'Accept': 'application/json', 'Content-Type': 'application/json'}
        self.amg_id, self.amg_obj = self.get_amg()

    def get_amg(self):
        endpoint = self.url + '/storage-systems/%s/async-mirrors' % self.ssid
        rc, amg_objs = request(endpoint, url_username=self.user, url_password=self.pwd, validate_certs=self.certs, headers=self.post_headers)
        try:
            amg_id = filter(lambda d: d['label'] == self.name, amg_objs)[0]['id']
            amg_obj = filter(lambda d: d['label'] == self.name, amg_objs)[0]
        except IndexError:
            self.module.fail_json(msg='There is no async mirror group  %s associated with storage array %s' % (self.name, self.ssid))
        return (amg_id, amg_obj)

    @property
    def current_state(self):
        amg_id, amg_obj = self.get_amg()
        return amg_obj['syncActivity']

    def run_sync_action(self):
        post_body = dict()
        if self.state == 'running':
            if self.current_state == 'idle':
                if self.delete_recovery_point:
                    post_body.update(dict(deleteRecoveryPointIfNecessary=self.delete_recovery_point))
                suffix = 'sync'
            else:
                suffix = 'resume'
        else:
            suffix = 'suspend'
        endpoint = self.url + '/storage-systems/%s/async-mirrors/%s/%s' % (self.ssid, self.amg_id, suffix)
        rc, resp = request(endpoint, method='POST', url_username=self.user, url_password=self.pwd, validate_certs=self.certs, data=json.dumps(post_body), headers=self.post_headers, ignore_errors=True)
        if not str(rc).startswith('2'):
            self.module.fail_json(msg=str(resp['errorMessage']))
        return resp

    def apply(self):
        state_map = dict(running=['active'], suspended=['userSuspended', 'internallySuspended', 'paused'], err=['unkown', '_UNDEFINED'])
        if self.current_state not in state_map[self.state]:
            if self.current_state in state_map['err']:
                self.module.fail_json(msg="The sync is a state of '%s', this requires manual intervention. " + 'Please investigate and try again' % self.current_state)
            else:
                self.amg_obj = self.run_sync_action()
        ret, amg = self.get_amg()
        self.module.exit_json(changed=False, **amg)