from __future__ import (absolute_import, division, print_function)
from ansible.plugins.action import ActionBase
from ansible.errors import AnsibleActionFail
from ansible_collections.cisco.meraki.plugins.plugin_utils.meraki import (
from ansible_collections.cisco.meraki.plugins.plugin_utils.exceptions import (
class OrganizationsAdmins(object):

    def __init__(self, params, meraki):
        self.meraki = meraki
        self.new_object = dict(email=params.get('email'), name=params.get('name'), orgAccess=params.get('orgAccess'), tags=params.get('tags'), networks=params.get('networks'), authenticationMethod=params.get('authenticationMethod'), organizationId=params.get('organizationId'), adminId=params.get('adminId'))

    def get_all_params(self, name=None, id=None):
        new_object_params = {}
        if self.new_object.get('organizationId') is not None or self.new_object.get('organization_id') is not None:
            new_object_params['organizationId'] = self.new_object.get('organizationId') or self.new_object.get('organization_id')
        return new_object_params

    def create_params(self):
        new_object_params = {}
        if self.new_object.get('email') is not None or self.new_object.get('email') is not None:
            new_object_params['email'] = self.new_object.get('email') or self.new_object.get('email')
        if self.new_object.get('name') is not None or self.new_object.get('name') is not None:
            new_object_params['name'] = self.new_object.get('name') or self.new_object.get('name')
        if self.new_object.get('orgAccess') is not None or self.new_object.get('org_access') is not None:
            new_object_params['orgAccess'] = self.new_object.get('orgAccess') or self.new_object.get('org_access')
        if self.new_object.get('tags') is not None or self.new_object.get('tags') is not None:
            new_object_params['tags'] = self.new_object.get('tags') or self.new_object.get('tags')
        if self.new_object.get('networks') is not None or self.new_object.get('networks') is not None:
            new_object_params['networks'] = self.new_object.get('networks') or self.new_object.get('networks')
        if self.new_object.get('authenticationMethod') is not None or self.new_object.get('authentication_method') is not None:
            new_object_params['authenticationMethod'] = self.new_object.get('authenticationMethod') or self.new_object.get('authentication_method')
        if self.new_object.get('organizationId') is not None or self.new_object.get('organization_id') is not None:
            new_object_params['organizationId'] = self.new_object.get('organizationId') or self.new_object.get('organization_id')
        return new_object_params

    def delete_by_id_params(self):
        new_object_params = {}
        if self.new_object.get('organizationId') is not None or self.new_object.get('organization_id') is not None:
            new_object_params['organizationId'] = self.new_object.get('organizationId') or self.new_object.get('organization_id')
        if self.new_object.get('adminId') is not None or self.new_object.get('admin_id') is not None:
            new_object_params['adminId'] = self.new_object.get('adminId') or self.new_object.get('admin_id')
        return new_object_params

    def update_by_id_params(self):
        new_object_params = {}
        if self.new_object.get('name') is not None or self.new_object.get('name') is not None:
            new_object_params['name'] = self.new_object.get('name') or self.new_object.get('name')
        if self.new_object.get('orgAccess') is not None or self.new_object.get('org_access') is not None:
            new_object_params['orgAccess'] = self.new_object.get('orgAccess') or self.new_object.get('org_access')
        if self.new_object.get('tags') is not None or self.new_object.get('tags') is not None:
            new_object_params['tags'] = self.new_object.get('tags') or self.new_object.get('tags')
        if self.new_object.get('networks') is not None or self.new_object.get('networks') is not None:
            new_object_params['networks'] = self.new_object.get('networks') or self.new_object.get('networks')
        if self.new_object.get('organizationId') is not None or self.new_object.get('organization_id') is not None:
            new_object_params['organizationId'] = self.new_object.get('organizationId') or self.new_object.get('organization_id')
        if self.new_object.get('adminId') is not None or self.new_object.get('admin_id') is not None:
            new_object_params['adminId'] = self.new_object.get('adminId') or self.new_object.get('admin_id')
        return new_object_params

    def get_object_by_name(self, name):
        result = None
        try:
            items = self.meraki.exec_meraki(family='organizations', function='getOrganizationAdmins', params=self.get_all_params(name=name))
            if isinstance(items, dict):
                if 'response' in items:
                    items = items.get('response')
            result = get_dict_result(items, 'email', name)
        except Exception as e:
            print('Error: ', e)
            result = None
        return result

    def get_object_by_id(self, id):
        result = None
        try:
            items = self.meraki.exec_meraki(family='organizations', function='getOrganizationAdmins', params=self.get_all_params(id=id))
            if isinstance(items, dict):
                if 'response' in items:
                    items = items.get('response')
            result = get_dict_result(items, 'id', id)
        except Exception as e:
            print('Error: ', e)
            result = None
        return result

    def exists(self):
        id_exists = False
        name_exists = False
        prev_obj = None
        o_id = self.new_object.get('id')
        o_id = o_id or self.new_object.get('admin_id') or self.new_object.get('adminId')
        name = self.new_object.get('email')
        if o_id:
            prev_obj = self.get_object_by_id(o_id)
            id_exists = prev_obj is not None and isinstance(prev_obj, dict)
        if not id_exists and name:
            prev_obj = self.get_object_by_name(name)
            name_exists = prev_obj is not None and isinstance(prev_obj, dict)
        if name_exists:
            _id = prev_obj.get('id')
            _id = _id or prev_obj.get('adminId')
            if id_exists and name_exists and (o_id != _id):
                raise InconsistentParameters("The 'id' and 'name' params don't refer to the same object")
            if _id:
                self.new_object.update(dict(id=_id))
                self.new_object.update(dict(adminId=_id))
        it_exists = prev_obj is not None and isinstance(prev_obj, dict)
        return (it_exists, prev_obj)

    def requires_update(self, current_obj):
        requested_obj = self.new_object
        obj_params = [('email', 'email'), ('name', 'name'), ('orgAccess', 'orgAccess'), ('tags', 'tags'), ('networks', 'networks'), ('authenticationMethod', 'authenticationMethod'), ('organizationId', 'organizationId'), ('adminId', 'adminId')]
        return any((not meraki_compare_equality(current_obj.get(meraki_param), requested_obj.get(ansible_param)) for meraki_param, ansible_param in obj_params))

    def create(self):
        result = self.meraki.exec_meraki(family='organizations', function='createOrganizationAdmin', params=self.create_params(), op_modifies=True)
        return result

    def update(self):
        id = self.new_object.get('id')
        id = id or self.new_object.get('adminId')
        name = self.new_object.get('email')
        result = None
        if not id:
            prev_obj_name = self.get_object_by_name(name)
            id_ = None
            if prev_obj_name:
                id_ = prev_obj_name.get('id')
                id_ = id_ or prev_obj_name.get('adminId')
            if id_:
                self.new_object.update(dict(adminId=id_))
        result = self.meraki.exec_meraki(family='organizations', function='updateOrganizationAdmin', params=self.update_by_id_params(), op_modifies=True)
        return result

    def delete(self):
        id = self.new_object.get('id')
        id = id or self.new_object.get('adminId')
        name = self.new_object.get('email')
        result = None
        if not id:
            prev_obj_name = self.get_object_by_name(name)
            id_ = None
            if prev_obj_name:
                id_ = prev_obj_name.get('id')
                id_ = id_ or prev_obj_name.get('adminId')
            if id_:
                self.new_object.update(dict(adminId=id_))
        result = self.meraki.exec_meraki(family='organizations', function='deleteOrganizationAdmin', params=self.delete_by_id_params())
        return result