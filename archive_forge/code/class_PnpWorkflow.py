from __future__ import (absolute_import, division, print_function)
from ansible.plugins.action import ActionBase
from ansible.errors import AnsibleActionFail
from ansible_collections.cisco.dnac.plugins.plugin_utils.dnac import (
from ansible_collections.cisco.dnac.plugins.plugin_utils.exceptions import (
class PnpWorkflow(object):

    def __init__(self, params, dnac):
        self.dnac = dnac
        self.new_object = dict(_id=params.get('_id'), addToInventory=params.get('addToInventory'), addedOn=params.get('addedOn'), configId=params.get('configId'), currTaskIdx=params.get('currTaskIdx'), description=params.get('description'), endTime=params.get('endTime'), execTime=params.get('execTime'), imageId=params.get('imageId'), instanceType=params.get('instanceType'), lastupdateOn=params.get('lastupdateOn'), name=params.get('name'), startTime=params.get('startTime'), state=params.get('state_'), tasks=params.get('tasks'), tenantId=params.get('tenantId'), type=params.get('type'), useState=params.get('useState'), version=params.get('version'), id=params.get('id'))

    def get_all_params(self, name=None, id=None):
        new_object_params = {}
        new_object_params['limit'] = self.new_object.get('limit')
        new_object_params['offset'] = self.new_object.get('offset')
        new_object_params['sort'] = self.new_object.get('sort')
        new_object_params['sort_order'] = self.new_object.get('sortOrder') or self.new_object.get('sort_order')
        new_object_params['type'] = self.new_object.get('type')
        new_object_params['name'] = name or self.new_object.get('name')
        return new_object_params

    def create_params(self):
        new_object_params = {}
        new_object_params['_id'] = self.new_object.get('_id')
        new_object_params['addToInventory'] = self.new_object.get('addToInventory')
        new_object_params['addedOn'] = self.new_object.get('addedOn')
        new_object_params['configId'] = self.new_object.get('configId')
        new_object_params['currTaskIdx'] = self.new_object.get('currTaskIdx')
        new_object_params['description'] = self.new_object.get('description')
        new_object_params['endTime'] = self.new_object.get('endTime')
        new_object_params['execTime'] = self.new_object.get('execTime')
        new_object_params['imageId'] = self.new_object.get('imageId')
        new_object_params['instanceType'] = self.new_object.get('instanceType')
        new_object_params['lastupdateOn'] = self.new_object.get('lastupdateOn')
        new_object_params['name'] = self.new_object.get('name')
        new_object_params['startTime'] = self.new_object.get('startTime')
        new_object_params['state_'] = self.new_object.get('state_')
        new_object_params['tasks'] = self.new_object.get('tasks')
        new_object_params['tenantId'] = self.new_object.get('tenantId')
        new_object_params['type'] = self.new_object.get('type')
        new_object_params['useState'] = self.new_object.get('useState')
        new_object_params['version'] = self.new_object.get('version')
        return new_object_params

    def delete_by_id_params(self):
        new_object_params = {}
        new_object_params['id'] = self.new_object.get('id')
        return new_object_params

    def update_by_id_params(self):
        new_object_params = {}
        new_object_params['_id'] = self.new_object.get('_id')
        new_object_params['addToInventory'] = self.new_object.get('addToInventory')
        new_object_params['addedOn'] = self.new_object.get('addedOn')
        new_object_params['configId'] = self.new_object.get('configId')
        new_object_params['currTaskIdx'] = self.new_object.get('currTaskIdx')
        new_object_params['description'] = self.new_object.get('description')
        new_object_params['endTime'] = self.new_object.get('endTime')
        new_object_params['execTime'] = self.new_object.get('execTime')
        new_object_params['imageId'] = self.new_object.get('imageId')
        new_object_params['instanceType'] = self.new_object.get('instanceType')
        new_object_params['lastupdateOn'] = self.new_object.get('lastupdateOn')
        new_object_params['name'] = self.new_object.get('name')
        new_object_params['startTime'] = self.new_object.get('startTime')
        new_object_params['state_'] = self.new_object.get('state_')
        new_object_params['tasks'] = self.new_object.get('tasks')
        new_object_params['tenantId'] = self.new_object.get('tenantId')
        new_object_params['type'] = self.new_object.get('type')
        new_object_params['useState'] = self.new_object.get('useState')
        new_object_params['version'] = self.new_object.get('version')
        new_object_params['id'] = self.new_object.get('id')
        return new_object_params

    def get_object_by_name(self, name):
        result = None
        try:
            items = self.dnac.exec(family='device_onboarding_pnp', function='get_workflows', params=self.get_all_params(name=name))
            if isinstance(items, dict):
                if 'response' in items:
                    items = items.get('response')
            result = get_dict_result(items, 'name', name)
        except Exception:
            result = None
        return result

    def get_object_by_id(self, id):
        result = None
        try:
            items = self.dnac.exec(family='device_onboarding_pnp', function='get_workflow_by_id', params={'id': id})
            if isinstance(items, dict):
                if 'response' in items:
                    items = items.get('response')
            result = get_dict_result(items, 'id', id)
        except Exception:
            result = None
        return result

    def exists(self):
        id_exists = False
        name_exists = False
        prev_obj = None
        o_id = self.new_object.get('id')
        name = self.new_object.get('name')
        if o_id:
            prev_obj = self.get_object_by_id(o_id)
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
            if _id:
                prev_obj = self.get_object_by_id(_id)
        it_exists = prev_obj is not None and isinstance(prev_obj, dict)
        return (it_exists, prev_obj)

    def requires_update(self, current_obj):
        requested_obj = self.new_object
        obj_params = [('_id', '_id'), ('addToInventory', 'addToInventory'), ('addedOn', 'addedOn'), ('configId', 'configId'), ('currTaskIdx', 'currTaskIdx'), ('description', 'description'), ('endTime', 'endTime'), ('execTime', 'execTime'), ('imageId', 'imageId'), ('instanceType', 'instanceType'), ('lastupdateOn', 'lastupdateOn'), ('name', 'name'), ('startTime', 'startTime'), ('state_', 'state'), ('tasks', 'tasks'), ('tenantId', 'tenantId'), ('type', 'type'), ('useState', 'useState'), ('version', 'version'), ('id', 'id')]
        return any((not dnac_compare_equality(current_obj.get(dnac_param), requested_obj.get(ansible_param)) for dnac_param, ansible_param in obj_params))

    def create(self):
        result = self.dnac.exec(family='device_onboarding_pnp', function='add_a_workflow', params=self.create_params(), op_modifies=True)
        return result

    def update(self):
        id = self.new_object.get('id')
        name = self.new_object.get('name')
        result = None
        if not id:
            prev_obj_name = self.get_object_by_name(name)
            id_ = None
            if prev_obj_name:
                id_ = prev_obj_name.get('id')
            if id_:
                self.new_object.update(dict(id=id_))
        result = self.dnac.exec(family='device_onboarding_pnp', function='update_workflow', params=self.update_by_id_params(), op_modifies=True)
        return result

    def delete(self):
        id = self.new_object.get('id')
        name = self.new_object.get('name')
        result = None
        if not id:
            prev_obj_name = self.get_object_by_name(name)
            id_ = None
            if prev_obj_name:
                id_ = prev_obj_name.get('id')
            if id_:
                self.new_object.update(dict(id=id_))
        result = self.dnac.exec(family='device_onboarding_pnp', function='delete_workflow_by_id', params=self.delete_by_id_params())
        return result