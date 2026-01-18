from __future__ import (absolute_import, division, print_function)
from ansible.plugins.action import ActionBase
from ansible.errors import AnsibleActionFail
from ansible_collections.cisco.meraki.plugins.plugin_utils.meraki import (
from ansible_collections.cisco.meraki.plugins.plugin_utils.exceptions import (
class OrganizationsPolicyObjects(object):

    def __init__(self, params, meraki):
        self.meraki = meraki
        self.new_object = dict(name=params.get('name'), category=params.get('category'), type=params.get('type'), cidr=params.get('cidr'), fqdn=params.get('fqdn'), mask=params.get('mask'), ip=params.get('ip'), groupIds=params.get('groupIds'), organizationId=params.get('organizationId'), policyObjectId=params.get('policyObjectId'))

    def get_all_params(self, name=None, id=None):
        new_object_params = {}
        if self.new_object.get('perPage') is not None or self.new_object.get('per_page') is not None:
            new_object_params['perPage'] = self.new_object.get('perPage') or self.new_object.get('per_page')
        new_object_params['total_pages'] = -1
        if self.new_object.get('startingAfter') is not None or self.new_object.get('starting_after') is not None:
            new_object_params['startingAfter'] = self.new_object.get('startingAfter') or self.new_object.get('starting_after')
        if self.new_object.get('endingBefore') is not None or self.new_object.get('ending_before') is not None:
            new_object_params['endingBefore'] = self.new_object.get('endingBefore') or self.new_object.get('ending_before')
        if self.new_object.get('organizationId') is not None or self.new_object.get('organization_id') is not None:
            new_object_params['organizationId'] = self.new_object.get('organizationId') or self.new_object.get('organization_id')
        return new_object_params

    def get_params_by_id(self, name=None, id=None):
        new_object_params = {}
        if self.new_object.get('organizationId') is not None or self.new_object.get('organization_id') is not None:
            new_object_params['organizationId'] = self.new_object.get('organizationId') or self.new_object.get('organization_id')
        if self.new_object.get('policyObjectId') is not None or self.new_object.get('policy_object_id') is not None:
            new_object_params['policyObjectId'] = self.new_object.get('policyObjectId') or self.new_object.get('policy_object_id')
        return new_object_params

    def create_params(self):
        new_object_params = {}
        if self.new_object.get('name') is not None or self.new_object.get('name') is not None:
            new_object_params['name'] = self.new_object.get('name') or self.new_object.get('name')
        if self.new_object.get('category') is not None or self.new_object.get('category') is not None:
            new_object_params['category'] = self.new_object.get('category') or self.new_object.get('category')
        if self.new_object.get('type') is not None or self.new_object.get('type') is not None:
            new_object_params['type'] = self.new_object.get('type') or self.new_object.get('type')
        if self.new_object.get('cidr') is not None or self.new_object.get('cidr') is not None:
            new_object_params['cidr'] = self.new_object.get('cidr') or self.new_object.get('cidr')
        if self.new_object.get('fqdn') is not None or self.new_object.get('fqdn') is not None:
            new_object_params['fqdn'] = self.new_object.get('fqdn') or self.new_object.get('fqdn')
        if self.new_object.get('mask') is not None or self.new_object.get('mask') is not None:
            new_object_params['mask'] = self.new_object.get('mask') or self.new_object.get('mask')
        if self.new_object.get('ip') is not None or self.new_object.get('ip') is not None:
            new_object_params['ip'] = self.new_object.get('ip') or self.new_object.get('ip')
        if self.new_object.get('groupIds') is not None or self.new_object.get('group_ids') is not None:
            new_object_params['groupIds'] = self.new_object.get('groupIds') or self.new_object.get('group_ids')
        if self.new_object.get('organizationId') is not None or self.new_object.get('organization_id') is not None:
            new_object_params['organizationId'] = self.new_object.get('organizationId') or self.new_object.get('organization_id')
        return new_object_params

    def delete_by_id_params(self):
        new_object_params = {}
        if self.new_object.get('organizationId') is not None or self.new_object.get('organization_id') is not None:
            new_object_params['organizationId'] = self.new_object.get('organizationId') or self.new_object.get('organization_id')
        if self.new_object.get('policyObjectId') is not None or self.new_object.get('policy_object_id') is not None:
            new_object_params['policyObjectId'] = self.new_object.get('policyObjectId') or self.new_object.get('policy_object_id')
        return new_object_params

    def update_by_id_params(self):
        new_object_params = {}
        if self.new_object.get('name') is not None or self.new_object.get('name') is not None:
            new_object_params['name'] = self.new_object.get('name') or self.new_object.get('name')
        if self.new_object.get('cidr') is not None or self.new_object.get('cidr') is not None:
            new_object_params['cidr'] = self.new_object.get('cidr') or self.new_object.get('cidr')
        if self.new_object.get('fqdn') is not None or self.new_object.get('fqdn') is not None:
            new_object_params['fqdn'] = self.new_object.get('fqdn') or self.new_object.get('fqdn')
        if self.new_object.get('mask') is not None or self.new_object.get('mask') is not None:
            new_object_params['mask'] = self.new_object.get('mask') or self.new_object.get('mask')
        if self.new_object.get('ip') is not None or self.new_object.get('ip') is not None:
            new_object_params['ip'] = self.new_object.get('ip') or self.new_object.get('ip')
        if self.new_object.get('groupIds') is not None or self.new_object.get('group_ids') is not None:
            new_object_params['groupIds'] = self.new_object.get('groupIds') or self.new_object.get('group_ids')
        if self.new_object.get('organizationId') is not None or self.new_object.get('organization_id') is not None:
            new_object_params['organizationId'] = self.new_object.get('organizationId') or self.new_object.get('organization_id')
        if self.new_object.get('policyObjectId') is not None or self.new_object.get('policy_object_id') is not None:
            new_object_params['policyObjectId'] = self.new_object.get('policyObjectId') or self.new_object.get('policy_object_id')
        return new_object_params

    def get_object_by_name(self, name):
        result = None
        try:
            items = self.meraki.exec_meraki(family='organizations', function='getOrganizationPolicyObjects', params=self.get_all_params(name=name))
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
            items = self.meraki.exec_meraki(family='organizations', function='getOrganizationPolicyObject', params=self.get_params_by_id())
            if isinstance(items, dict):
                if 'response' in items:
                    items = items.get('response')
            result = get_dict_result(items, 'policyObjectId', id)
        except Exception as e:
            print('Error: ', e)
            result = None
        return result

    def exists(self):
        id_exists = False
        name_exists = False
        prev_obj = None
        o_id = self.new_object.get('id')
        o_id = o_id or self.new_object.get('policy_object_id') or self.new_object.get('policyObjectId')
        name = self.new_object.get('name')
        if o_id:
            prev_obj = self.get_object_by_id(o_id)
            id_exists = prev_obj is not None and isinstance(prev_obj, dict)
        if not id_exists and name:
            prev_obj = self.get_object_by_name(name)
            name_exists = prev_obj is not None and isinstance(prev_obj, dict)
        if name_exists:
            _id = prev_obj.get('id')
            _id = _id or prev_obj.get('policyObjectId')
            if id_exists and name_exists and (o_id != _id):
                raise InconsistentParameters("The 'id' and 'name' params don't refer to the same object")
            if _id:
                self.new_object.update(dict(id=_id))
                self.new_object.update(dict(policyObjectId=_id))
            if _id:
                prev_obj = self.get_object_by_id(_id)
        it_exists = prev_obj is not None and isinstance(prev_obj, dict)
        return (it_exists, prev_obj)

    def requires_update(self, current_obj):
        requested_obj = self.new_object
        obj_params = [('name', 'name'), ('category', 'category'), ('type', 'type'), ('cidr', 'cidr'), ('fqdn', 'fqdn'), ('mask', 'mask'), ('ip', 'ip'), ('groupIds', 'groupIds'), ('organizationId', 'organizationId'), ('policyObjectId', 'policyObjectId')]
        return any((not meraki_compare_equality(current_obj.get(meraki_param), requested_obj.get(ansible_param)) for meraki_param, ansible_param in obj_params))

    def create(self):
        result = self.meraki.exec_meraki(family='organizations', function='createOrganizationPolicyObject', params=self.create_params(), op_modifies=True)
        return result

    def update(self):
        id = self.new_object.get('id')
        id = id or self.new_object.get('policyObjectId')
        name = self.new_object.get('name')
        result = None
        if not id:
            prev_obj_name = self.get_object_by_name(name)
            id_ = None
            if prev_obj_name:
                id_ = prev_obj_name.get('id')
                id_ = id_ or prev_obj_name.get('policyObjectId')
            if id_:
                self.new_object.update(dict(policyObjectId=id_))
        result = self.meraki.exec_meraki(family='organizations', function='updateOrganizationPolicyObject', params=self.update_by_id_params(), op_modifies=True)
        return result

    def delete(self):
        id = self.new_object.get('id')
        id = id or self.new_object.get('policyObjectId')
        name = self.new_object.get('name')
        result = None
        if not id:
            prev_obj_name = self.get_object_by_name(name)
            id_ = None
            if prev_obj_name:
                id_ = prev_obj_name.get('id')
                id_ = id_ or prev_obj_name.get('policyObjectId')
            if id_:
                self.new_object.update(dict(policyObjectId=id_))
        result = self.meraki.exec_meraki(family='organizations', function='deleteOrganizationPolicyObject', params=self.delete_by_id_params())
        return result