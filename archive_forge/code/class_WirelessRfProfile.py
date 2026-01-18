from __future__ import (absolute_import, division, print_function)
from ansible.plugins.action import ActionBase
from ansible.errors import AnsibleActionFail
from ansible_collections.cisco.dnac.plugins.plugin_utils.dnac import (
from ansible_collections.cisco.dnac.plugins.plugin_utils.exceptions import (
class WirelessRfProfile(object):

    def __init__(self, params, dnac):
        self.dnac = dnac
        self.new_object = dict(name=params.get('name'), defaultRfProfile=params.get('defaultRfProfile'), enableRadioTypeA=params.get('enableRadioTypeA'), enableRadioTypeB=params.get('enableRadioTypeB'), channelWidth=params.get('channelWidth'), enableCustom=params.get('enableCustom'), enableBrownField=params.get('enableBrownField'), radioTypeAProperties=params.get('radioTypeAProperties'), radioTypeBProperties=params.get('radioTypeBProperties'), radioTypeCProperties=params.get('radioTypeCProperties'), enableRadioTypeC=params.get('enableRadioTypeC'), rf_profile_name=params.get('rfProfileName'))

    def get_all_params(self, name=None, id=None):
        new_object_params = {}
        new_object_params['rf_profile_name'] = self.new_object.get('rf_profile_name') or self.new_object.get('name')
        return new_object_params

    def create_params(self):
        new_object_params = {}
        new_object_params['name'] = self.new_object.get('rf_profile_name') or self.new_object.get('name')
        new_object_params['defaultRfProfile'] = self.new_object.get('defaultRfProfile')
        new_object_params['enableRadioTypeA'] = self.new_object.get('enableRadioTypeA')
        new_object_params['enableRadioTypeB'] = self.new_object.get('enableRadioTypeB')
        new_object_params['channelWidth'] = self.new_object.get('channelWidth')
        new_object_params['enableCustom'] = self.new_object.get('enableCustom')
        new_object_params['enableBrownField'] = self.new_object.get('enableBrownField')
        new_object_params['radioTypeAProperties'] = self.new_object.get('radioTypeAProperties')
        new_object_params['radioTypeBProperties'] = self.new_object.get('radioTypeBProperties')
        new_object_params['radioTypeCProperties'] = self.new_object.get('radioTypeCProperties')
        new_object_params['enableRadioTypeC'] = self.new_object.get('enableRadioTypeC')
        return new_object_params

    def delete_by_name_params(self):
        new_object_params = {}
        new_object_params['rf_profile_name'] = self.new_object.get('rf_profile_name') or self.new_object.get('name')
        return new_object_params

    def get_object_by_name(self, name):
        result = None
        try:
            items = self.dnac.exec(family='wireless', function='retrieve_rf_profiles', params=self.get_all_params(name=name))
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
        o_id = self.new_object.get('id')
        name = self.new_object.get('name')
        name = name or self.new_object.get('rf_profile_name')
        if o_id:
            prev_obj = self.get_object_by_id(o_id)
            id_exists = prev_obj is not None and isinstance(prev_obj, dict)
        if not id_exists and name:
            prev_obj = self.get_object_by_name(name)
            name_exists = prev_obj is not None and isinstance(prev_obj, dict)
        if id_exists:
            _name = prev_obj.get('name')
            _name = _name or prev_obj.get('rfProfileName')
            if _name:
                self.new_object.update(dict(rf_profile_name=_name))
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
        obj_params = [('name', 'name'), ('defaultRfProfile', 'defaultRfProfile'), ('enableRadioTypeA', 'enableRadioTypeA'), ('enableRadioTypeB', 'enableRadioTypeB'), ('channelWidth', 'channelWidth'), ('enableCustom', 'enableCustom'), ('enableBrownField', 'enableBrownField'), ('radioTypeAProperties', 'radioTypeAProperties'), ('radioTypeBProperties', 'radioTypeBProperties'), ('radioTypeCProperties', 'radioTypeCProperties'), ('enableRadioTypeC', 'enableRadioTypeC'), ('rfProfileName', 'rf_profile_name')]
        return any((not dnac_compare_equality(current_obj.get(dnac_param), requested_obj.get(ansible_param)) for dnac_param, ansible_param in obj_params))

    def create(self):
        result = self.dnac.exec(family='wireless', function='create_or_update_rf_profile', params=self.create_params(), op_modifies=True)
        return result

    def delete(self):
        id = self.new_object.get('id')
        name = self.new_object.get('name')
        name = name or self.new_object.get('rf_profile_name')
        result = None
        if not name:
            prev_obj_id = self.get_object_by_id(id)
            name_ = None
            if prev_obj_id:
                name_ = prev_obj_id.get('name')
                name_ = name_ or prev_obj_id.get('rfProfileName')
            if name_:
                self.new_object.update(dict(rf_profile_name=name_))
        result = self.dnac.exec(family='wireless', function='delete_rf_profiles', params=self.delete_by_name_params())
        return result