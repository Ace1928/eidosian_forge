from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (AnsibleCloudStack, cs_argument_spec,
class AnsibleCloudStackUser(AnsibleCloudStack):

    def __init__(self, module):
        super(AnsibleCloudStackUser, self).__init__(module)
        self.returns = {'username': 'username', 'firstname': 'first_name', 'lastname': 'last_name', 'email': 'email', 'secretkey': 'user_api_secret', 'apikey': 'user_api_key', 'timezone': 'timezone'}
        self.account_types = {'user': 0, 'root_admin': 1, 'domain_admin': 2}
        self.user = None

    def get_account_type(self):
        account_type = self.module.params.get('account_type')
        return self.account_types[account_type]

    def get_user(self):
        if not self.user:
            args = {'domainid': self.get_domain('id'), 'fetch_list': True}
            users = self.query_api('listUsers', **args)
            if users:
                user_name = self.module.params.get('username')
                for u in users:
                    if user_name.lower() == u['username'].lower():
                        self.user = u
                        break
        return self.user

    def enable_user(self):
        user = self.get_user()
        if not user:
            user = self.present_user()
        if user['state'].lower() != 'enabled':
            self.result['changed'] = True
            args = {'id': user['id']}
            if not self.module.check_mode:
                res = self.query_api('enableUser', **args)
                user = res['user']
        return user

    def lock_user(self):
        user = self.get_user()
        if not user:
            user = self.present_user()
        if user['state'].lower() == 'disabled':
            user = self.enable_user()
        if user['state'].lower() != 'locked':
            self.result['changed'] = True
            args = {'id': user['id']}
            if not self.module.check_mode:
                res = self.query_api('lockUser', **args)
                user = res['user']
        return user

    def disable_user(self):
        user = self.get_user()
        if not user:
            user = self.present_user()
        if user['state'].lower() != 'disabled':
            self.result['changed'] = True
            args = {'id': user['id']}
            if not self.module.check_mode:
                user = self.query_api('disableUser', **args)
                poll_async = self.module.params.get('poll_async')
                if poll_async:
                    user = self.poll_job(user, 'user')
        return user

    def present_user(self):
        required_params = ['account', 'email', 'password', 'first_name', 'last_name']
        self.module.fail_on_missing_params(required_params=required_params)
        user = self.get_user()
        if user:
            user = self._update_user(user)
        else:
            user = self._create_user(user)
        return user

    def _get_common_args(self):
        return {'firstname': self.module.params.get('first_name'), 'lastname': self.module.params.get('last_name'), 'email': self.module.params.get('email'), 'timezone': self.module.params.get('timezone')}

    def _create_user(self, user):
        self.result['changed'] = True
        args = self._get_common_args()
        args.update({'account': self.get_account(key='name'), 'domainid': self.get_domain('id'), 'username': self.module.params.get('username'), 'password': self.module.params.get('password')})
        if not self.module.check_mode:
            res = self.query_api('createUser', **args)
            user = res['user']
            if self.module.params.get('keys_registered'):
                res = self.query_api('registerUserKeys', id=user['id'])
                user.update(res['userkeys'])
        return user

    def _update_user(self, user):
        args = self._get_common_args()
        args.update({'id': user['id']})
        if self.has_changed(args, user):
            self.result['changed'] = True
            if not self.module.check_mode:
                res = self.query_api('updateUser', **args)
                user = res['user']
        if 'apikey' not in user and self.module.params.get('keys_registered'):
            self.result['changed'] = True
            if not self.module.check_mode:
                res = self.query_api('registerUserKeys', id=user['id'])
                user.update(res['userkeys'])
        return user

    def absent_user(self):
        user = self.get_user()
        if user:
            self.result['changed'] = True
            if not self.module.check_mode:
                self.query_api('deleteUser', id=user['id'])
        return user

    def get_result(self, resource):
        super(AnsibleCloudStackUser, self).get_result(resource)
        if resource:
            if 'accounttype' in resource:
                for key, value in self.account_types.items():
                    if value == resource['accounttype']:
                        self.result['account_type'] = key
                        break
            if self.module.params.get('keys_registered') and 'apikey' in resource and ('secretkey' not in resource):
                user_keys = self.query_api('getUserKeys', id=resource['id'])
                if user_keys:
                    self.result['user_api_secret'] = user_keys['userkeys'].get('secretkey')
        return self.result