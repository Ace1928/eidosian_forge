from __future__ import absolute_import, division, print_function
from ansible.plugins.action import ActionBase
from ansible.errors import AnsibleActionFail
from ansible_collections.cisco.ise.plugins.plugin_utils.ise import (
from ansible_collections.cisco.ise.plugins.plugin_utils.exceptions import (
class RadiusServerSequence(object):

    def __init__(self, params, ise):
        self.ise = ise
        self.new_object = dict(name=params.get('name'), description=params.get('description'), strip_prefix=params.get('stripPrefix'), strip_suffix=params.get('stripSuffix'), prefix_separator=params.get('prefixSeparator'), suffix_separator=params.get('suffixSeparator'), remote_accounting=params.get('remoteAccounting'), local_accounting=params.get('localAccounting'), use_attr_set_on_request=params.get('useAttrSetOnRequest'), use_attr_set_before_acc=params.get('useAttrSetBeforeAcc'), continue_authorz_policy=params.get('continueAuthorzPolicy'), radius_server_list=params.get('RadiusServerList'), on_request_attr_manipulator_list=params.get('OnRequestAttrManipulatorList'), before_accept_attr_manipulators_list=params.get('BeforeAcceptAttrManipulatorsList'), id=params.get('id'))

    def get_object_by_name(self, name):
        result = None
        gen_items_responses = self.ise.exec(family='radius_server_sequence', function='get_radius_server_sequence_generator')
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
            result = self.ise.exec(family='radius_server_sequence', function='get_radius_server_sequence_by_id', handle_func_exception=False, params={'id': id}).response['RadiusServerSequence']
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
        obj_params = [('name', 'name'), ('description', 'description'), ('stripPrefix', 'strip_prefix'), ('stripSuffix', 'strip_suffix'), ('prefixSeparator', 'prefix_separator'), ('suffixSeparator', 'suffix_separator'), ('remoteAccounting', 'remote_accounting'), ('localAccounting', 'local_accounting'), ('useAttrSetOnRequest', 'use_attr_set_on_request'), ('useAttrSetBeforeAcc', 'use_attr_set_before_acc'), ('continueAuthorzPolicy', 'continue_authorz_policy'), ('RadiusServerList', 'radius_server_list'), ('OnRequestAttrManipulatorList', 'on_request_attr_manipulator_list'), ('BeforeAcceptAttrManipulatorsList', 'before_accept_attr_manipulators_list'), ('id', 'id')]
        return any((not ise_compare_equality(current_obj.get(ise_param), requested_obj.get(ansible_param)) for ise_param, ansible_param in obj_params))

    def create(self):
        result = self.ise.exec(family='radius_server_sequence', function='create_radius_server_sequence', params=self.new_object).response
        return result

    def update(self):
        id = self.new_object.get('id')
        name = self.new_object.get('name')
        result = None
        if not id:
            id_ = self.get_object_by_name(name).get('id')
            self.new_object.update(dict(id=id_))
        result = self.ise.exec(family='radius_server_sequence', function='update_radius_server_sequence_by_id', params=self.new_object).response
        return result

    def delete(self):
        id = self.new_object.get('id')
        name = self.new_object.get('name')
        result = None
        if not id:
            id_ = self.get_object_by_name(name).get('id')
            self.new_object.update(dict(id=id_))
        result = self.ise.exec(family='radius_server_sequence', function='delete_radius_server_sequence_by_id', params=self.new_object).response
        return result