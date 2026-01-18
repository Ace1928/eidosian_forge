from __future__ import (absolute_import, division, print_function)
from ansible.plugins.action import ActionBase
from ansible.errors import AnsibleActionFail
from ansible_collections.cisco.meraki.plugins.plugin_utils.meraki import (
from ansible_collections.cisco.meraki.plugins.plugin_utils.exceptions import (
class OrganizationsLoginSecurity(object):

    def __init__(self, params, meraki):
        self.meraki = meraki
        self.new_object = dict(enforcePasswordExpiration=params.get('enforcePasswordExpiration'), passwordExpirationDays=params.get('passwordExpirationDays'), enforceDifferentPasswords=params.get('enforceDifferentPasswords'), numDifferentPasswords=params.get('numDifferentPasswords'), enforceStrongPasswords=params.get('enforceStrongPasswords'), enforceAccountLockout=params.get('enforceAccountLockout'), accountLockoutAttempts=params.get('accountLockoutAttempts'), enforceIdleTimeout=params.get('enforceIdleTimeout'), idleTimeoutMinutes=params.get('idleTimeoutMinutes'), enforceTwoFactorAuth=params.get('enforceTwoFactorAuth'), enforceLoginIpRanges=params.get('enforceLoginIpRanges'), loginIpRanges=params.get('loginIpRanges'), apiAuthentication=params.get('apiAuthentication'), organization_id=params.get('organizationId'))

    def get_all_params(self, name=None, id=None):
        new_object_params = {}
        if self.new_object.get('organizationId') is not None or self.new_object.get('organization_id') is not None:
            new_object_params['organizationId'] = self.new_object.get('organizationId') or self.new_object.get('organization_id')
        return new_object_params

    def update_all_params(self):
        new_object_params = {}
        if self.new_object.get('enforcePasswordExpiration') is not None or self.new_object.get('enforce_password_expiration') is not None:
            new_object_params['enforcePasswordExpiration'] = self.new_object.get('enforcePasswordExpiration')
        if self.new_object.get('passwordExpirationDays') is not None or self.new_object.get('password_expiration_days') is not None:
            new_object_params['passwordExpirationDays'] = self.new_object.get('passwordExpirationDays') or self.new_object.get('password_expiration_days')
        if self.new_object.get('enforceDifferentPasswords') is not None or self.new_object.get('enforce_different_passwords') is not None:
            new_object_params['enforceDifferentPasswords'] = self.new_object.get('enforceDifferentPasswords')
        if self.new_object.get('numDifferentPasswords') is not None or self.new_object.get('num_different_passwords') is not None:
            new_object_params['numDifferentPasswords'] = self.new_object.get('numDifferentPasswords') or self.new_object.get('num_different_passwords')
        if self.new_object.get('enforceStrongPasswords') is not None or self.new_object.get('enforce_strong_passwords') is not None:
            new_object_params['enforceStrongPasswords'] = self.new_object.get('enforceStrongPasswords')
        if self.new_object.get('enforceAccountLockout') is not None or self.new_object.get('enforce_account_lockout') is not None:
            new_object_params['enforceAccountLockout'] = self.new_object.get('enforceAccountLockout')
        if self.new_object.get('accountLockoutAttempts') is not None or self.new_object.get('account_lockout_attempts') is not None:
            new_object_params['accountLockoutAttempts'] = self.new_object.get('accountLockoutAttempts') or self.new_object.get('account_lockout_attempts')
        if self.new_object.get('enforceIdleTimeout') is not None or self.new_object.get('enforce_idle_timeout') is not None:
            new_object_params['enforceIdleTimeout'] = self.new_object.get('enforceIdleTimeout')
        if self.new_object.get('idleTimeoutMinutes') is not None or self.new_object.get('idle_timeout_minutes') is not None:
            new_object_params['idleTimeoutMinutes'] = self.new_object.get('idleTimeoutMinutes') or self.new_object.get('idle_timeout_minutes')
        if self.new_object.get('enforceTwoFactorAuth') is not None or self.new_object.get('enforce_two_factor_auth') is not None:
            new_object_params['enforceTwoFactorAuth'] = self.new_object.get('enforceTwoFactorAuth')
        if self.new_object.get('enforceLoginIpRanges') is not None or self.new_object.get('enforce_login_ip_ranges') is not None:
            new_object_params['enforceLoginIpRanges'] = self.new_object.get('enforceLoginIpRanges')
        if self.new_object.get('loginIpRanges') is not None or self.new_object.get('login_ip_ranges') is not None:
            new_object_params['loginIpRanges'] = self.new_object.get('loginIpRanges') or self.new_object.get('login_ip_ranges')
        if self.new_object.get('apiAuthentication') is not None or self.new_object.get('api_authentication') is not None:
            new_object_params['apiAuthentication'] = self.new_object.get('apiAuthentication') or self.new_object.get('api_authentication')
        if self.new_object.get('organizationId') is not None or self.new_object.get('organization_id') is not None:
            new_object_params['organizationId'] = self.new_object.get('organizationId') or self.new_object.get('organization_id')
        return new_object_params

    def get_object_by_name(self, name):
        result = None
        try:
            items = self.meraki.exec_meraki(family='organizations', function='getOrganizationLoginSecurity', params=self.get_all_params(name=name))
            if isinstance(items, dict):
                if 'response' in items:
                    items = items.get('response')
            result = get_dict_result(items, 'name', name)
            if result is None:
                result = items
        except Exception as e:
            print('Error: ', e)
            result = None
        return result

    def get_object_by_id(self, id):
        result = None
        return result

    def exists(self):
        prev_obj = None
        id_exists = False
        name_exists = False
        o_id = self.new_object.get('id')
        name = self.new_object.get('organization_id')
        if o_id:
            prev_obj = self.get_object_by_name(o_id)
            id_exists = prev_obj is not None and isinstance(prev_obj, dict)
        if not id_exists and name:
            prev_obj = self.get_object_by_name(name)
            name_exists = prev_obj is not None and isinstance(prev_obj, dict)
        if name_exists:
            _id = prev_obj.get('id')
            if id_exists and name_exists and (o_id != _id):
                raise InconsistentParameters("The 'id' and 'name' params don't refer to the same object")
            if _id:
                self.new_object.update(dict(id=_id))
        it_exists = prev_obj is not None and isinstance(prev_obj, dict)
        return (it_exists, prev_obj)

    def requires_update(self, current_obj):
        requested_obj = self.new_object
        obj_params = [('enforcePasswordExpiration', 'enforcePasswordExpiration'), ('enforceDifferentPasswords', 'enforceDifferentPasswords'), ('numDifferentPasswords', 'numDifferentPasswords'), ('enforceStrongPasswords', 'enforceStrongPasswords'), ('enforceAccountLockout', 'enforceAccountLockout'), ('accountLockoutAttempts', 'accountLockoutAttempts'), ('enforceIdleTimeout', 'enforceIdleTimeout'), ('idleTimeoutMinutes', 'idleTimeoutMinutes'), ('enforceTwoFactorAuth', 'enforceTwoFactorAuth'), ('enforceLoginIpRanges', 'enforceLoginIpRanges'), ('loginIpRanges', 'loginIpRanges'), ('apiAuthentication', 'apiAuthentication'), ('organizationId', 'organizationId')]
        return any((not meraki_compare_equality(current_obj.get(meraki_param), requested_obj.get(ansible_param)) for meraki_param, ansible_param in obj_params))

    def update(self):
        id = self.new_object.get('id')
        name = self.new_object.get('name')
        result = None
        result = self.meraki.exec_meraki(family='organizations', function='updateOrganizationLoginSecurity', params=self.update_all_params(), op_modifies=True)
        return result