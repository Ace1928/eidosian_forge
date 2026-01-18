from __future__ import (absolute_import, division, print_function)
from ansible.plugins.action import ActionBase
from ansible.errors import AnsibleActionFail
from ansible_collections.cisco.meraki.plugins.plugin_utils.meraki import (
from ansible_collections.cisco.meraki.plugins.plugin_utils.exceptions import (
class NetworksSensorAlertsProfiles(object):

    def __init__(self, params, meraki):
        self.meraki = meraki
        self.new_object = dict(name=params.get('name'), schedule=params.get('schedule'), conditions=params.get('conditions'), recipients=params.get('recipients'), serials=params.get('serials'), networkId=params.get('networkId'), id=params.get('id'))

    def get_all_params(self, name=None, id=None):
        new_object_params = {}
        if self.new_object.get('networkId') is not None or self.new_object.get('network_id') is not None:
            new_object_params['networkId'] = self.new_object.get('networkId') or self.new_object.get('network_id')
        return new_object_params

    def get_params_by_id(self, name=None, id=None):
        new_object_params = {}
        if self.new_object.get('networkId') is not None or self.new_object.get('network_id') is not None:
            new_object_params['networkId'] = self.new_object.get('networkId') or self.new_object.get('network_id')
        if self.new_object.get('id') is not None or self.new_object.get('id') is not None:
            new_object_params['id'] = self.new_object.get('id')
        return new_object_params

    def create_params(self):
        new_object_params = {}
        if self.new_object.get('name') is not None or self.new_object.get('name') is not None:
            new_object_params['name'] = self.new_object.get('name') or self.new_object.get('name')
        if self.new_object.get('schedule') is not None or self.new_object.get('schedule') is not None:
            new_object_params['schedule'] = self.new_object.get('schedule') or self.new_object.get('schedule')
        if self.new_object.get('conditions') is not None or self.new_object.get('conditions') is not None:
            new_object_params['conditions'] = self.new_object.get('conditions') or self.new_object.get('conditions')
        if self.new_object.get('recipients') is not None or self.new_object.get('recipients') is not None:
            new_object_params['recipients'] = self.new_object.get('recipients') or self.new_object.get('recipients')
        if self.new_object.get('serials') is not None or self.new_object.get('serials') is not None:
            new_object_params['serials'] = self.new_object.get('serials') or self.new_object.get('serials')
        if self.new_object.get('networkId') is not None or self.new_object.get('network_id') is not None:
            new_object_params['networkId'] = self.new_object.get('networkId') or self.new_object.get('network_id')
        return new_object_params

    def delete_by_id_params(self):
        new_object_params = {}
        if self.new_object.get('networkId') is not None or self.new_object.get('network_id') is not None:
            new_object_params['networkId'] = self.new_object.get('networkId') or self.new_object.get('network_id')
        if self.new_object.get('id') is not None or self.new_object.get('id') is not None:
            new_object_params['id'] = self.new_object.get('id') or self.new_object.get('id')
        return new_object_params

    def update_by_id_params(self):
        new_object_params = {}
        if self.new_object.get('name') is not None or self.new_object.get('name') is not None:
            new_object_params['name'] = self.new_object.get('name') or self.new_object.get('name')
        if self.new_object.get('schedule') is not None or self.new_object.get('schedule') is not None:
            new_object_params['schedule'] = self.new_object.get('schedule') or self.new_object.get('schedule')
        if self.new_object.get('conditions') is not None or self.new_object.get('conditions') is not None:
            new_object_params['conditions'] = self.new_object.get('conditions') or self.new_object.get('conditions')
        if self.new_object.get('recipients') is not None or self.new_object.get('recipients') is not None:
            new_object_params['recipients'] = self.new_object.get('recipients') or self.new_object.get('recipients')
        if self.new_object.get('serials') is not None or self.new_object.get('serials') is not None:
            new_object_params['serials'] = self.new_object.get('serials') or self.new_object.get('serials')
        if self.new_object.get('networkId') is not None or self.new_object.get('network_id') is not None:
            new_object_params['networkId'] = self.new_object.get('networkId') or self.new_object.get('network_id')
        if self.new_object.get('id') is not None or self.new_object.get('id') is not None:
            new_object_params['id'] = self.new_object.get('id') or self.new_object.get('id')
        return new_object_params

    def get_object_by_name(self, name):
        result = None
        try:
            items = self.meraki.exec_meraki(family='sensor', function='getNetworkSensorAlertsProfiles', params=self.get_all_params(name=name))
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
            items = self.meraki.exec_meraki(family='sensor', function='getNetworkSensorAlertsProfile', params={'id': id})
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
        obj_params = [('name', 'name'), ('schedule', 'schedule'), ('conditions', 'conditions'), ('recipients', 'recipients'), ('serials', 'serials'), ('networkId', 'networkId'), ('id', 'id')]
        return any((not meraki_compare_equality(current_obj.get(meraki_param), requested_obj.get(ansible_param)) for meraki_param, ansible_param in obj_params))

    def create(self):
        result = self.meraki.exec_meraki(family='sensor', function='createNetworkSensorAlertsProfile', params=self.create_params(), op_modifies=True)
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
        result = self.meraki.exec_meraki(family='sensor', function='updateNetworkSensorAlertsProfile', params=self.update_by_id_params(), op_modifies=True)
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
        result = self.meraki.exec_meraki(family='sensor', function='deleteNetworkSensorAlertsProfile', params=self.delete_by_id_params())
        return result