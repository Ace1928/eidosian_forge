from __future__ import absolute_import, division, print_function
from ansible.plugins.action import ActionBase
from ansible.errors import AnsibleActionFail
from ansible_collections.cisco.ise.plugins.plugin_utils.ise import (
from ansible_collections.cisco.ise.plugins.plugin_utils.exceptions import (
class Sgt(object):

    def __init__(self, params, ise):
        self.ise = ise
        self.new_object = dict(name=params.get('name'), description=params.get('description'), value=params.get('value'), generation_id=params.get('generationId'), is_read_only=params.get('isReadOnly'), propogate_to_apic=params.get('propogateToApic'), default_sgacls=params.get('defaultSGACLs'), id=params.get('id'))

    def get_object_by_name(self, name):
        result = None
        gen_items_responses = self.ise.exec(family='security_groups', function='get_security_groups_generator')
        try:
            for items_response in gen_items_responses:
                items = items_response.response['SearchResult']['resources']
                result = get_dict_result(items, 'name', name)
                if result:
                    return result
        except (TypeError, AttributeError) as e:
            self.ise.fail_json(msg='An error occured when executing operation. Check the configuration of your API Settings and API Gateway settings on your ISE server. This collection assumes that the API Gateway, the ERS APIs and OpenAPIs are enabled. You may want to enable the (ise_debug: True) argument. The error was: {error}'.format(error=e))
        except Exception:
            result = None
            return result
        return result

    def get_object_by_id(self, id):
        try:
            result = self.ise.exec(family='security_groups', function='get_security_group_by_id', handle_func_exception=False, params={'id': id}).response['Sgt']
        except (TypeError, AttributeError) as e:
            self.ise.fail_json(msg='An error occured when executing operation. Check the configuration of your API Settings and API Gateway settings on your ISE server. This collection assumes that the API Gateway, the ERS APIs and OpenAPIs are enabled. You may want to enable the (ise_debug: True) argument. The error was: {error}'.format(error=e))
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
                prev_obj = self.get_object_by_id(_id)
        it_exists = prev_obj is not None and isinstance(prev_obj, dict)
        return (it_exists, prev_obj)

    def requires_update(self, current_obj):
        requested_obj = self.new_object
        if requested_obj.get('value') and requested_obj.get('value') == -1:
            requested_obj['value'] = current_obj.get('value')
        obj_params = [('name', 'name'), ('description', 'description'), ('value', 'value'), ('generationId', 'generation_id'), ('isReadOnly', 'is_read_only'), ('propogateToApic', 'propogate_to_apic'), ('defaultSGACLs', 'default_sgacls'), ('id', 'id')]
        return any((not ise_compare_equality(current_obj.get(ise_param), requested_obj.get(ansible_param)) for ise_param, ansible_param in obj_params))

    def create(self):
        result = self.ise.exec(family='security_groups', function='create_security_group', params=self.new_object).response
        return result

    def update(self):
        id = self.new_object.get('id')
        name = self.new_object.get('name')
        result = None
        if not id:
            id_ = self.get_object_by_name(name).get('id')
            self.new_object.update(dict(id=id_))
        result = self.ise.exec(family='security_groups', function='update_security_group_by_id', params=self.new_object).response
        return result

    def delete(self):
        id = self.new_object.get('id')
        name = self.new_object.get('name')
        result = None
        if not id:
            id_ = self.get_object_by_name(name).get('id')
            self.new_object.update(dict(id=id_))
        result = self.ise.exec(family='security_groups', function='delete_security_group_by_id', params=self.new_object).response
        return result