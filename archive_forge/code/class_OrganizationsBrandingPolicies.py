from __future__ import (absolute_import, division, print_function)
from ansible.plugins.action import ActionBase
from ansible.errors import AnsibleActionFail
from ansible_collections.cisco.meraki.plugins.plugin_utils.meraki import (
from ansible_collections.cisco.meraki.plugins.plugin_utils.exceptions import (
class OrganizationsBrandingPolicies(object):

    def __init__(self, params, meraki):
        self.meraki = meraki
        self.new_object = dict(name=params.get('name'), enabled=params.get('enabled'), adminSettings=params.get('adminSettings'), helpSettings=params.get('helpSettings'), customLogo=params.get('customLogo'), organizationId=params.get('organizationId'), brandingPolicyId=params.get('brandingPolicyId'))

    def get_all_params(self, name=None, id=None):
        new_object_params = {}
        if self.new_object.get('organizationId') is not None or self.new_object.get('organization_id') is not None:
            new_object_params['organizationId'] = self.new_object.get('organizationId') or self.new_object.get('organization_id')
        return new_object_params

    def get_params_by_id(self, name=None, id=None):
        new_object_params = {}
        if self.new_object.get('organizationId') is not None or self.new_object.get('organization_id') is not None:
            new_object_params['organizationId'] = self.new_object.get('organizationId') or self.new_object.get('organization_id')
        if self.new_object.get('brandingPolicyId') is not None or self.new_object.get('branding_policy_id') is not None:
            new_object_params['brandingPolicyId'] = self.new_object.get('brandingPolicyId') or self.new_object.get('branding_policy_id')
        return new_object_params

    def create_params(self):
        new_object_params = {}
        if self.new_object.get('name') is not None or self.new_object.get('name') is not None:
            new_object_params['name'] = self.new_object.get('name') or self.new_object.get('name')
        if self.new_object.get('enabled') is not None or self.new_object.get('enabled') is not None:
            new_object_params['enabled'] = self.new_object.get('enabled')
        if self.new_object.get('adminSettings') is not None or self.new_object.get('admin_settings') is not None:
            new_object_params['adminSettings'] = self.new_object.get('adminSettings') or self.new_object.get('admin_settings')
        if self.new_object.get('helpSettings') is not None or self.new_object.get('help_settings') is not None:
            new_object_params['helpSettings'] = self.new_object.get('helpSettings') or self.new_object.get('help_settings')
        if self.new_object.get('customLogo') is not None or self.new_object.get('custom_logo') is not None:
            new_object_params['customLogo'] = self.new_object.get('customLogo') or self.new_object.get('custom_logo')
        if self.new_object.get('organizationId') is not None or self.new_object.get('organization_id') is not None:
            new_object_params['organizationId'] = self.new_object.get('organizationId') or self.new_object.get('organization_id')
        return new_object_params

    def delete_by_id_params(self):
        new_object_params = {}
        if self.new_object.get('organizationId') is not None or self.new_object.get('organization_id') is not None:
            new_object_params['organizationId'] = self.new_object.get('organizationId') or self.new_object.get('organization_id')
        if self.new_object.get('brandingPolicyId') is not None or self.new_object.get('branding_policy_id') is not None:
            new_object_params['brandingPolicyId'] = self.new_object.get('brandingPolicyId') or self.new_object.get('branding_policy_id')
        return new_object_params

    def update_by_id_params(self):
        new_object_params = {}
        if self.new_object.get('name') is not None or self.new_object.get('name') is not None:
            new_object_params['name'] = self.new_object.get('name') or self.new_object.get('name')
        if self.new_object.get('enabled') is not None or self.new_object.get('enabled') is not None:
            new_object_params['enabled'] = self.new_object.get('enabled')
        if self.new_object.get('adminSettings') is not None or self.new_object.get('admin_settings') is not None:
            new_object_params['adminSettings'] = self.new_object.get('adminSettings') or self.new_object.get('admin_settings')
        if self.new_object.get('helpSettings') is not None or self.new_object.get('help_settings') is not None:
            new_object_params['helpSettings'] = self.new_object.get('helpSettings') or self.new_object.get('help_settings')
        if self.new_object.get('customLogo') is not None or self.new_object.get('custom_logo') is not None:
            new_object_params['customLogo'] = self.new_object.get('customLogo') or self.new_object.get('custom_logo')
        if self.new_object.get('organizationId') is not None or self.new_object.get('organization_id') is not None:
            new_object_params['organizationId'] = self.new_object.get('organizationId') or self.new_object.get('organization_id')
        if self.new_object.get('brandingPolicyId') is not None or self.new_object.get('branding_policy_id') is not None:
            new_object_params['brandingPolicyId'] = self.new_object.get('brandingPolicyId') or self.new_object.get('branding_policy_id')
        return new_object_params

    def get_object_by_name(self, name):
        result = None
        try:
            items = self.meraki.exec_meraki(family='organizations', function='getOrganizationBrandingPolicies', params=self.get_all_params(name=name))
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
        try:
            items = self.meraki.exec_meraki(family='organizations', function='getOrganizationBrandingPolicy', params=self.get_params_by_id())
            if isinstance(items, dict):
                if 'response' in items:
                    items = items.get('response')
            result = get_dict_result(items, 'brandingPolicyId', id)
        except Exception as e:
            print('Error: ', e)
            result = None
        return result

    def exists(self):
        id_exists = False
        name_exists = False
        prev_obj = None
        o_id = self.new_object.get('id')
        o_id = o_id or self.new_object.get('branding_policy_id') or self.new_object.get('brandingPolicyId')
        name = self.new_object.get('name')
        if o_id:
            prev_obj = self.get_object_by_id(o_id)
            id_exists = prev_obj is not None and isinstance(prev_obj, dict)
        if not id_exists and name:
            prev_obj = self.get_object_by_name(name)
            name_exists = prev_obj is not None and isinstance(prev_obj, dict)
        if name_exists:
            _id = prev_obj.get('id')
            _id = _id or prev_obj.get('brandingPolicyId')
            if id_exists and name_exists and (o_id != _id):
                raise InconsistentParameters("The 'id' and 'name' params don't refer to the same object")
            if _id:
                self.new_object.update(dict(id=_id))
                self.new_object.update(dict(brandingPolicyId=_id))
            if _id:
                prev_obj = self.get_object_by_id(_id)
        it_exists = prev_obj is not None and isinstance(prev_obj, dict)
        return (it_exists, prev_obj)

    def requires_update(self, current_obj):
        requested_obj = self.new_object
        obj_params = [('name', 'name'), ('enabled', 'enabled'), ('adminSettings', 'adminSettings'), ('helpSettings', 'helpSettings'), ('customLogo', 'customLogo'), ('organizationId', 'organizationId'), ('brandingPolicyId', 'brandingPolicyId')]
        return any((not meraki_compare_equality(current_obj.get(meraki_param), requested_obj.get(ansible_param)) for meraki_param, ansible_param in obj_params))

    def create(self):
        result = self.meraki.exec_meraki(family='organizations', function='createOrganizationBrandingPolicy', params=self.create_params(), op_modifies=True)
        return result

    def update(self):
        id = self.new_object.get('id')
        id = id or self.new_object.get('brandingPolicyId')
        name = self.new_object.get('name')
        result = None
        if not id:
            prev_obj_name = self.get_object_by_name(name)
            id_ = None
            if prev_obj_name:
                id_ = prev_obj_name.get('id')
                id_ = id_ or prev_obj_name.get('brandingPolicyId')
            if id_:
                self.new_object.update(dict(brandingPolicyId=id_))
        result = self.meraki.exec_meraki(family='organizations', function='updateOrganizationBrandingPolicy', params=self.update_by_id_params(), op_modifies=True)
        return result

    def delete(self):
        id = self.new_object.get('id')
        id = id or self.new_object.get('brandingPolicyId')
        name = self.new_object.get('name')
        result = None
        if not id:
            prev_obj_name = self.get_object_by_name(name)
            id_ = None
            if prev_obj_name:
                id_ = prev_obj_name.get('id')
                id_ = id_ or prev_obj_name.get('brandingPolicyId')
            if id_:
                self.new_object.update(dict(brandingPolicyId=id_))
        result = self.meraki.exec_meraki(family='organizations', function='deleteOrganizationBrandingPolicy', params=self.delete_by_id_params())
        return result