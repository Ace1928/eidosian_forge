from __future__ import absolute_import, division, print_function
from ansible.plugins.action import ActionBase
from ansible.errors import AnsibleActionFail
from ansible_collections.cisco.ise.plugins.plugin_utils.ise import (
from ansible_collections.cisco.ise.plugins.plugin_utils.exceptions import (
class DeviceAdministrationTimeDateConditions(object):

    def __init__(self, params, ise):
        self.ise = ise
        self.new_object = dict(condition_type=params.get('conditionType'), is_negate=params.get('isNegate'), link=params.get('link'), description=params.get('description'), id=params.get('id'), name=params.get('name'), attribute_name=params.get('attributeName'), attribute_id=params.get('attributeId'), attribute_value=params.get('attributeValue'), dictionary_name=params.get('dictionaryName'), dictionary_value=params.get('dictionaryValue'), operator=params.get('operator'), children=params.get('children'), dates_range=params.get('datesRange'), dates_range_exception=params.get('datesRangeException'), hours_range=params.get('hoursRange'), hours_range_exception=params.get('hoursRangeException'), week_days=params.get('weekDays'), week_days_exception=params.get('weekDaysException'))

    def get_object_by_name(self, name):
        result = None
        items = self.ise.exec(family='device_administration_time_date_conditions', function='get_device_admin_time_conditions').response.get('response', []) or []
        result = get_dict_result(items, 'name', name)
        return result

    def get_object_by_id(self, id):
        try:
            result = self.ise.exec(family='device_administration_time_date_conditions', function='get_device_admin_time_condition_by_id', params={'id': id}, handle_func_exception=False).response['response']
        except (TypeError, AttributeError) as e:
            self.ise.fail_json(msg='An error occured when executing operation. Check the configuration of your API Settings and API Gateway settings on your ISE server. This collection assumes that the API Gateway, the ERS APIs and OpenAPIs are enabled. You may want to enable the (ise_debug: True) argument. The error was: {error}'.format(error=e))
        except Exception:
            result = None
        return result

    def exists(self):
        prev_obj = None
        id_exists = False
        name_exists = False
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
                prev_obj = self.get_object_by_id(_id)
        it_exists = prev_obj is not None and isinstance(prev_obj, dict)
        return (it_exists, prev_obj)

    def requires_update(self, current_obj):
        requested_obj = self.new_object
        obj_params = [('conditionType', 'condition_type'), ('isNegate', 'is_negate'), ('link', 'link'), ('description', 'description'), ('id', 'id'), ('name', 'name'), ('attributeName', 'attribute_name'), ('attributeId', 'attribute_id'), ('attributeValue', 'attribute_value'), ('dictionaryName', 'dictionary_name'), ('dictionaryValue', 'dictionary_value'), ('operator', 'operator'), ('children', 'children'), ('datesRange', 'dates_range'), ('datesRangeException', 'dates_range_exception'), ('hoursRange', 'hours_range'), ('hoursRangeException', 'hours_range_exception'), ('weekDays', 'week_days'), ('weekDaysException', 'week_days_exception')]
        return any((not ise_compare_equality(current_obj.get(ise_param), requested_obj.get(ansible_param)) for ise_param, ansible_param in obj_params))

    def create(self):
        result = self.ise.exec(family='device_administration_time_date_conditions', function='create_device_admin_time_condition', params=self.new_object).response
        return result

    def update(self):
        id = self.new_object.get('id')
        name = self.new_object.get('name')
        result = None
        if not id:
            id_ = self.get_object_by_name(name).get('id')
            self.new_object.update(dict(id=id_))
        result = self.ise.exec(family='device_administration_time_date_conditions', function='update_device_admin_time_condition_by_id', params=self.new_object).response
        return result

    def delete(self):
        id = self.new_object.get('id')
        name = self.new_object.get('name')
        result = None
        if not id:
            id_ = self.get_object_by_name(name).get('id')
            self.new_object.update(dict(id=id_))
        result = self.ise.exec(family='device_administration_time_date_conditions', function='delete_device_admin_time_condition_by_id', params=self.new_object).response
        return result