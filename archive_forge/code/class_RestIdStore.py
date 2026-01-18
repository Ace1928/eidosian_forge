from __future__ import absolute_import, division, print_function
from ansible.plugins.action import ActionBase
from ansible.errors import AnsibleActionFail
from ansible_collections.cisco.ise.plugins.plugin_utils.ise import (
class RestIdStore(object):

    def __init__(self, params, ise):
        self.ise = ise
        self.new_object = dict(name=params.get('name'), description=params.get('description'), ers_rest_idstore_attributes=params.get('ersRestIDStoreAttributes'), ers_rest_idstore_user_attributes=params.get('ersRestIDStoreUserAttributes'), id=params.get('id'))

    def get_object_by_name(self, name):
        try:
            result = self.ise.exec(family='restid_store', function='get_rest_id_store_by_name', params={'name': name}, handle_func_exception=False).response['ERSRestIDStore']
            result = get_dict_result(result, 'name', name)
        except (TypeError, AttributeError) as e:
            self.ise.fail_json(msg='An error occured when executing operation. Check the configuration of your API Settings and API Gateway settings on your ISE server. This collection assumes that the API Gateway, the ERS APIs and OpenAPIs are enabled. You may want to enable the (ise_debug: True) argument. The error was: {error}'.format(error=e))
        except Exception:
            result = None
        return result

    def get_object_by_id(self, id):
        try:
            result = self.ise.exec(family='restid_store', function='get_rest_id_store_by_id', handle_func_exception=False, params={'id': id}).response['ERSRestIDStore']
        except (TypeError, AttributeError) as e:
            self.ise.fail_json(msg='An error occured when executing operation. Check the configuration of your API Settings and API Gateway settings on your ISE server. This collection assumes that the API Gateway, the ERS APIs and OpenAPIs are enabled. You may want to enable the (ise_debug: True) argument. The error was: {error}'.format(error=e))
        except Exception:
            result = None
        return result

    def exists(self):
        result = False
        prev_obj = None
        id = self.new_object.get('id')
        name = self.new_object.get('name')
        if id:
            prev_obj = self.get_object_by_id(id)
            result = prev_obj is not None and isinstance(prev_obj, dict)
        elif name:
            prev_obj = self.get_object_by_name(name)
            result = prev_obj is not None and isinstance(prev_obj, dict)
        return (result, prev_obj)

    def requires_update(self, current_obj):
        requested_obj = self.new_object
        obj_params = [('name', 'name'), ('description', 'description'), ('ersRestIDStoreAttributes', 'ers_rest_idstore_attributes'), ('ersRestIDStoreUserAttributes', 'ers_rest_idstore_user_attributes'), ('id', 'id')]
        return any((not ise_compare_equality(current_obj.get(ise_param), requested_obj.get(ansible_param)) for ise_param, ansible_param in obj_params))

    def create(self):
        print(self.new_object)
        result = self.ise.exec(family='restid_store', function='create_rest_id_store', params=self.new_object).response
        return result

    def update(self):
        id = self.new_object.get('id')
        name = self.new_object.get('name')
        result = None
        if id:
            result = self.ise.exec(family='restid_store', function='update_rest_id_store_by_id', params=self.new_object).response
        elif name:
            result = self.ise.exec(family='restid_store', function='update_rest_id_store_by_name', params=self.new_object).response
        return result

    def delete(self):
        id = self.new_object.get('id')
        name = self.new_object.get('name')
        result = None
        if id:
            result = self.ise.exec(family='restid_store', function='delete_rest_id_store_by_id', params=self.new_object).response
        elif name:
            result = self.ise.exec(family='restid_store', function='delete_rest_id_store_by_name', params=self.new_object).response
        return result