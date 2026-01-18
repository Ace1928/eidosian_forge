from __future__ import absolute_import, division, print_function
import json
from ansible.module_utils.basic import AnsibleModule, env_fallback
from ansible.module_utils.six.moves.urllib.parse import urlencode
from ansible.module_utils.urls import fetch_url
class StackiHost(object):

    def __init__(self, module):
        self.module = module
        self.hostname = module.params['name']
        self.rack = module.params['rack']
        self.rank = module.params['rank']
        self.appliance = module.params['appliance']
        self.prim_intf = module.params['prim_intf']
        self.prim_intf_ip = module.params['prim_intf_ip']
        self.network = module.params['network']
        self.prim_intf_mac = module.params['prim_intf_mac']
        self.endpoint = module.params['stacki_endpoint']
        auth_creds = {'USERNAME': module.params['stacki_user'], 'PASSWORD': module.params['stacki_password']}
        cred_a = self.do_request(self.endpoint, method='GET')
        cookie_a = cred_a.headers.get('Set-Cookie').split(';')
        init_csrftoken = None
        for c in cookie_a:
            if 'csrftoken' in c:
                init_csrftoken = c.replace('csrftoken=', '')
                init_csrftoken = init_csrftoken.rstrip('\r\n')
                break
        header = {'csrftoken': init_csrftoken, 'X-CSRFToken': init_csrftoken, 'Content-type': 'application/x-www-form-urlencoded', 'Cookie': cred_a.headers.get('Set-Cookie')}
        login_endpoint = self.endpoint + '/login'
        login_req = self.do_request(login_endpoint, headers=header, payload=urlencode(auth_creds), method='POST')
        cookie_f = login_req.headers.get('Set-Cookie').split(';')
        csrftoken = None
        for f in cookie_f:
            if 'csrftoken' in f:
                csrftoken = f.replace('csrftoken=', '')
            if 'sessionid' in f:
                sessionid = c.split('sessionid=', 1)[-1]
                sessionid = sessionid.rstrip('\r\n')
        self.header = {'csrftoken': csrftoken, 'X-CSRFToken': csrftoken, 'sessionid': sessionid, 'Content-type': 'application/json', 'Cookie': login_req.headers.get('Set-Cookie')}

    def do_request(self, url, payload=None, headers=None, method=None):
        res, info = fetch_url(self.module, url, data=payload, headers=headers, method=method)
        if info['status'] != 200:
            self.module.fail_json(changed=False, msg=info['msg'])
        return res

    def stack_check_host(self):
        res = self.do_request(self.endpoint, payload=json.dumps({'cmd': 'list host'}), headers=self.header, method='POST')
        return self.hostname in res.read()

    def stack_sync(self):
        self.do_request(self.endpoint, payload=json.dumps({'cmd': 'sync config'}), headers=self.header, method='POST')
        self.do_request(self.endpoint, payload=json.dumps({'cmd': 'sync host config'}), headers=self.header, method='POST')

    def stack_force_install(self, result):
        data = {'cmd': 'set host boot {0} action=install'.format(self.hostname)}
        self.do_request(self.endpoint, payload=json.dumps(data), headers=self.header, method='POST')
        changed = True
        self.stack_sync()
        result['changed'] = changed
        result['stdout'] = 'api call successful'.rstrip('\r\n')

    def stack_add(self, result):
        data = dict()
        changed = False
        data['cmd'] = 'add host {0} rack={1} rank={2} appliance={3}'.format(self.hostname, self.rack, self.rank, self.appliance)
        self.do_request(self.endpoint, payload=json.dumps(data), headers=self.header, method='POST')
        self.stack_sync()
        result['changed'] = changed
        result['stdout'] = 'api call successful'.rstrip('\r\n')

    def stack_remove(self, result):
        data = dict()
        data['cmd'] = 'remove host {0}'.format(self.hostname)
        self.do_request(self.endpoint, payload=json.dumps(data), headers=self.header, method='POST')
        self.stack_sync()
        result['changed'] = True
        result['stdout'] = 'api call successful'.rstrip('\r\n')