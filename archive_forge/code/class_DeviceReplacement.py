from __future__ import (absolute_import, division, print_function)
from ansible.plugins.action import ActionBase
from ansible.errors import AnsibleActionFail
from ansible_collections.cisco.dnac.plugins.plugin_utils.dnac import (
from ansible_collections.cisco.dnac.plugins.plugin_utils.exceptions import (
class DeviceReplacement(object):

    def __init__(self, params, dnac):
        self.dnac = dnac
        self.new_object = dict(payload=params.get('payload'))

    def get_all_params(self, name=None, id=None):
        new_object_params = {}
        new_object_params['faulty_device_name'] = self.new_object.get('faultyDeviceName') or self.new_object.get('faulty_device_name')
        new_object_params['faulty_device_platform'] = self.new_object.get('faultyDevicePlatform') or self.new_object.get('faulty_device_platform')
        new_object_params['replacement_device_platform'] = self.new_object.get('replacementDevicePlatform') or self.new_object.get('replacement_device_platform')
        new_object_params['faulty_device_serial_number'] = self.new_object.get('faultyDeviceSerialNumber') or self.new_object.get('faulty_device_serial_number')
        new_object_params['replacement_device_serial_number'] = self.new_object.get('replacementDeviceSerialNumber') or self.new_object.get('replacement_device_serial_number')
        new_object_params['replacement_status'] = self.new_object.get('replacementStatus') or self.new_object.get('replacement_status')
        new_object_params['family'] = self.new_object.get('family')
        new_object_params['sort_by'] = self.new_object.get('sortBy') or self.new_object.get('sort_by')
        new_object_params['sort_order'] = self.new_object.get('sortOrder') or self.new_object.get('sort_order')
        new_object_params['offset'] = self.new_object.get('offset')
        new_object_params['limit'] = self.new_object.get('limit')
        return new_object_params

    def create_params(self):
        new_object_params = {}
        new_object_params['payload'] = self.new_object.get('payload')
        return new_object_params

    def update_all_params(self):
        new_object_params = {}
        new_object_params['payload'] = self.new_object.get('payload')
        return new_object_params

    def get_object_by_name(self, name):
        result = None
        try:
            items = self.dnac.exec(family='device_replacement', function='return_replacement_devices_with_details', params=self.get_all_params(name=name))
            if isinstance(items, dict):
                if 'response' in items:
                    items = items.get('response')
            result = get_dict_result(items, 'name', name)
        except Exception:
            result = None
        return result

    def get_object_by_id(self, id):
        result = None
        return result

    def exists(self):
        prev_obj = None
        id_exists = False
        name_exists = False
        requested_obj = self.new_object.get('payload')
        if requested_obj and len(requested_obj) > 0:
            requested_obj = requested_obj[0]
        o_id = self.new_object.get('id') or requested_obj.get('id')
        name = self.new_object.get('name') or requested_obj.get('name')
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
        it_exists = prev_obj is not None and isinstance(prev_obj, dict)
        return (it_exists, prev_obj)

    def requires_update(self, current_obj):
        requested_obj = self.new_object.get('payload')
        if requested_obj and len(requested_obj) > 0:
            requested_obj = requested_obj[0]
        obj_params = [('creationTime', 'creationTime'), ('family', 'family'), ('faultyDeviceId', 'faultyDeviceId'), ('faultyDeviceName', 'faultyDeviceName'), ('faultyDevicePlatform', 'faultyDevicePlatform'), ('faultyDeviceSerialNumber', 'faultyDeviceSerialNumber'), ('id', 'id'), ('neighbourDeviceId', 'neighbourDeviceId'), ('networkReadinessTaskId', 'networkReadinessTaskId'), ('replacementDevicePlatform', 'replacementDevicePlatform'), ('replacementDeviceSerialNumber', 'replacementDeviceSerialNumber'), ('replacementStatus', 'replacementStatus'), ('replacementTime', 'replacementTime'), ('workflowId', 'workflowId')]
        return any((not dnac_compare_equality(current_obj.get(dnac_param), requested_obj.get(ansible_param)) for dnac_param, ansible_param in obj_params))

    def create(self):
        result = self.dnac.exec(family='device_replacement', function='mark_device_for_replacement', params=self.create_params(), op_modifies=True)
        return result

    def update(self):
        requested_obj = self.new_object.get('payload')
        if requested_obj and len(requested_obj) > 0:
            requested_obj = requested_obj[0]
        id = self.new_object.get('id') or requested_obj.get('id')
        name = self.new_object.get('name') or requested_obj.get('name')
        result = None
        result = self.dnac.exec(family='device_replacement', function='unmark_device_for_replacement', params=self.update_all_params(), op_modifies=True)
        return result