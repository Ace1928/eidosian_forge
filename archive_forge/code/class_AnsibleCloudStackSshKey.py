from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils._text import to_native
from ..module_utils.cloudstack import (
class AnsibleCloudStackSshKey(AnsibleCloudStack):

    def __init__(self, module):
        super(AnsibleCloudStackSshKey, self).__init__(module)
        self.returns = {'privatekey': 'private_key', 'fingerprint': 'fingerprint'}
        self.ssh_key = None

    def register_ssh_key(self, public_key):
        ssh_key = self.get_ssh_key()
        args = self._get_common_args()
        name = self.module.params.get('name')
        res = None
        if not ssh_key:
            self.result['changed'] = True
            args['publickey'] = public_key
            if not self.module.check_mode:
                args['name'] = name
                res = self.query_api('registerSSHKeyPair', **args)
        else:
            fingerprint = self._get_ssh_fingerprint(public_key)
            if ssh_key['fingerprint'] != fingerprint:
                self.result['changed'] = True
                if not self.module.check_mode:
                    args['name'] = name
                    self.query_api('deleteSSHKeyPair', **args)
            elif ssh_key['name'].lower() != name.lower():
                self.result['changed'] = True
                if not self.module.check_mode:
                    args['name'] = ssh_key['name']
                    self.query_api('deleteSSHKeyPair', **args)
                    self.ssh_key = None
                    ssh_key = self.get_ssh_key()
                    if ssh_key and ssh_key['fingerprint'] != fingerprint:
                        args['name'] = name
                        self.query_api('deleteSSHKeyPair', **args)
            if not self.module.check_mode and self.result['changed']:
                args['publickey'] = public_key
                args['name'] = name
                res = self.query_api('registerSSHKeyPair', **args)
        if res and 'keypair' in res:
            ssh_key = res['keypair']
        return ssh_key

    def create_ssh_key(self):
        ssh_key = self.get_ssh_key()
        if not ssh_key:
            self.result['changed'] = True
            args = self._get_common_args()
            args['name'] = self.module.params.get('name')
            if not self.module.check_mode:
                res = self.query_api('createSSHKeyPair', **args)
                ssh_key = res['keypair']
        return ssh_key

    def remove_ssh_key(self, name=None):
        ssh_key = self.get_ssh_key()
        if ssh_key:
            self.result['changed'] = True
            args = self._get_common_args()
            args['name'] = name or self.module.params.get('name')
            if not self.module.check_mode:
                self.query_api('deleteSSHKeyPair', **args)
        return ssh_key

    def _get_common_args(self):
        return {'domainid': self.get_domain('id'), 'account': self.get_account('name'), 'projectid': self.get_project('id')}

    def get_ssh_key(self):
        if not self.ssh_key:
            public_key = self.module.params.get('public_key')
            if public_key:
                args_fingerprint = self._get_common_args()
                args_fingerprint['fingerprint'] = self._get_ssh_fingerprint(public_key)
                ssh_keys = self.query_api('listSSHKeyPairs', **args_fingerprint)
                if ssh_keys and 'sshkeypair' in ssh_keys:
                    self.ssh_key = ssh_keys['sshkeypair'][0]
            if not self.ssh_key:
                args_name = self._get_common_args()
                args_name['name'] = self.module.params.get('name')
                ssh_keys = self.query_api('listSSHKeyPairs', **args_name)
                if ssh_keys and 'sshkeypair' in ssh_keys:
                    self.ssh_key = ssh_keys['sshkeypair'][0]
        return self.ssh_key

    def _get_ssh_fingerprint(self, public_key):
        key = sshpubkeys.SSHKey(public_key)
        if hasattr(key, 'hash_md5'):
            return key.hash_md5().replace(to_native('MD5:'), to_native(''))
        return key.hash()