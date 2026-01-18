from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (AnsibleCloudStack, cs_argument_spec,
class AnsibleCloudStackAccount(AnsibleCloudStack):

    def __init__(self, module):
        super(AnsibleCloudStackAccount, self).__init__(module)
        self.returns = {'networkdomain': 'network_domain', 'rolename': 'role'}
        self.account = None
        self.account_types = {'user': 0, 'root_admin': 1, 'domain_admin': 2}

    def get_role_id(self):
        role_param = self.module.params.get('role')
        role_id = None
        if role_param:
            role_list = self.query_api('listRoles')
            for role in role_list['role']:
                if role_param in [role['name'], role['id']]:
                    role_id = role['id']
            if not role_id:
                self.module.fail_json(msg='Role not found: %s' % role_param)
        return role_id

    def get_account_type(self):
        account_type = self.module.params.get('account_type')
        return self.account_types[account_type]

    def get_account(self):
        if not self.account:
            args = {'listall': True, 'domainid': self.get_domain(key='id'), 'fetch_list': True}
            accounts = self.query_api('listAccounts', **args)
            if accounts:
                account_name = self.module.params.get('name')
                for a in accounts:
                    if account_name == a['name']:
                        self.account = a
                        break
        return self.account

    def enable_account(self):
        account = self.get_account()
        if not account:
            account = self.present_account()
        if account['state'].lower() != 'enabled':
            self.result['changed'] = True
            args = {'id': account['id'], 'account': self.module.params.get('name'), 'domainid': self.get_domain(key='id')}
            if not self.module.check_mode:
                res = self.query_api('enableAccount', **args)
                account = res['account']
        return account

    def lock_account(self):
        return self.lock_or_disable_account(lock=True)

    def disable_account(self):
        return self.lock_or_disable_account()

    def lock_or_disable_account(self, lock=False):
        account = self.get_account()
        if not account:
            account = self.present_account()
        if lock and account['state'].lower() == 'disabled':
            account = self.enable_account()
        if lock and account['state'].lower() != 'locked' or (not lock and account['state'].lower() != 'disabled'):
            self.result['changed'] = True
            args = {'id': account['id'], 'account': self.module.params.get('name'), 'domainid': self.get_domain(key='id'), 'lock': lock}
            if not self.module.check_mode:
                account = self.query_api('disableAccount', **args)
                poll_async = self.module.params.get('poll_async')
                if poll_async:
                    account = self.poll_job(account, 'account')
        return account

    def present_account(self):
        account = self.get_account()
        if not account:
            self.result['changed'] = True
            if self.module.params.get('ldap_domain'):
                required_params = ['domain', 'username']
                self.module.fail_on_missing_params(required_params=required_params)
                account = self.create_ldap_account(account)
            else:
                required_params = ['email', 'username', 'password', 'first_name', 'last_name']
                self.module.fail_on_missing_params(required_params=required_params)
                account = self.create_account(account)
        return account

    def create_ldap_account(self, account):
        args = {'account': self.module.params.get('name'), 'domainid': self.get_domain(key='id'), 'accounttype': self.get_account_type(), 'networkdomain': self.module.params.get('network_domain'), 'username': self.module.params.get('username'), 'timezone': self.module.params.get('timezone'), 'roleid': self.get_role_id()}
        if not self.module.check_mode:
            res = self.query_api('ldapCreateAccount', **args)
            account = res['account']
            args = {'account': self.module.params.get('name'), 'domainid': self.get_domain(key='id'), 'accounttype': self.get_account_type(), 'ldapdomain': self.module.params.get('ldap_domain'), 'type': self.module.params.get('ldap_type')}
            self.query_api('linkAccountToLdap', **args)
        return account

    def create_account(self, account):
        args = {'account': self.module.params.get('name'), 'domainid': self.get_domain(key='id'), 'accounttype': self.get_account_type(), 'networkdomain': self.module.params.get('network_domain'), 'username': self.module.params.get('username'), 'password': self.module.params.get('password'), 'firstname': self.module.params.get('first_name'), 'lastname': self.module.params.get('last_name'), 'email': self.module.params.get('email'), 'timezone': self.module.params.get('timezone'), 'roleid': self.get_role_id()}
        if not self.module.check_mode:
            res = self.query_api('createAccount', **args)
            account = res['account']
        return account

    def absent_account(self):
        account = self.get_account()
        if account:
            self.result['changed'] = True
            if not self.module.check_mode:
                res = self.query_api('deleteAccount', id=account['id'])
                poll_async = self.module.params.get('poll_async')
                if poll_async:
                    self.poll_job(res, 'account')
        return account

    def get_result(self, resource):
        super(AnsibleCloudStackAccount, self).get_result(resource)
        if resource:
            if 'accounttype' in resource:
                for key, value in self.account_types.items():
                    if value == resource['accounttype']:
                        self.result['account_type'] = key
                        break
        return self.result