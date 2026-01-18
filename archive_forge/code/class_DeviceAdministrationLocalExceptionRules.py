from __future__ import absolute_import, division, print_function
from ansible.plugins.action import ActionBase
from ansible.errors import AnsibleActionFail
from ansible_collections.cisco.ise.plugins.plugin_utils.ise import (
from ansible_collections.cisco.ise.plugins.plugin_utils.exceptions import (
class DeviceAdministrationLocalExceptionRules(object):

    def __init__(self, params, ise):
        self.ise = ise
        self.new_object = dict(commands=params.get('commands'), link=params.get('link'), profile=params.get('profile'), rule=params.get('rule'), policy_id=params.get('policyId'), id=params.get('id'))

    def get_object_by_name(self, name, policy_id):
        result = None
        items = self.ise.exec(family='device_administration_authorization_exception_rules', function='get_device_admin_local_exception_rules', params={'policy_id': policy_id}).response.get('response', []) or []
        for item in items:
            if item.get('rule') and item['rule'].get('name') == name and item['rule'].get('id'):
                result = dict(item)
                return result
        return result

    def get_object_by_id(self, id, policy_id):
        try:
            result = self.ise.exec(family='device_administration_authorization_exception_rules', function='get_device_admin_local_exception_rule_by_id', handle_func_exception=False, params={'id': id, 'policy_id': policy_id}).response['response']
        except (TypeError, AttributeError) as e:
            self.ise.fail_json(msg='An error occured when executing operation. Check the configuration of your API Settings and API Gateway settings on your ISE server. This collection assumes that the API Gateway, the ERS APIs and OpenAPIs are enabled. You may want to enable the (ise_debug: True) argument. The error was: {error}'.format(error=e))
        except Exception:
            result = None
        return result

    def exists(self):
        prev_obj = None
        id_exists = False
        name_exists = False
        name = False
        o_id = self.new_object.get('id')
        policy_id = self.new_object.get('policy_id')
        if self.new_object.get('rule', {}) is not None:
            name = self.new_object.get('rule', {}).get('name')
            o_id = o_id or self.new_object.get('rule', {}).get('id')
        if o_id:
            prev_obj = self.get_object_by_id(o_id, policy_id)
            id_exists = prev_obj is not None and isinstance(prev_obj, dict)
        if not id_exists and name:
            prev_obj = self.get_object_by_name(name, policy_id)
            name_exists = prev_obj is not None and isinstance(prev_obj, dict)
        if name_exists:
            _id = prev_obj.get('rule', {}).get('id')
            if id_exists and name_exists and (o_id != _id):
                raise InconsistentParameters("The 'id' and 'name' params don't refer to the same object")
            if _id:
                prev_obj = self.get_object_by_id(_id, policy_id)
        it_exists = prev_obj is not None and isinstance(prev_obj, dict)
        return (it_exists, prev_obj)

    def requires_update(self, current_obj):
        requested_obj = self.new_object
        obj_params = [('commands', 'commands'), ('link', 'link'), ('profile', 'profile'), ('rule', 'rule'), ('policyId', 'policy_id'), ('id', 'id')]
        return any((not ise_compare_equality(current_obj.get(ise_param), requested_obj.get(ansible_param)) for ise_param, ansible_param in obj_params))

    def create(self):
        result = self.ise.exec(family='device_administration_authorization_exception_rules', function='create_device_admin_local_exception_rule', params=self.new_object).response
        return result

    def update(self):
        id = self.new_object.get('id')
        name = False
        if self.new_object.get('rule', {}) is not None:
            name = self.new_object.get('rule', {}).get('name')
            id = id or self.new_object.get('rule', {}).get('id')
        policy_id = self.new_object.get('policy_id')
        result = None
        if not id:
            id_ = self.get_object_by_name(name, policy_id).get('rule', {}).get('id')
            rule = self.new_object.get('rule', {})
            rule.update(dict(id=id_))
            self.new_object.update(dict(rule=rule, id=id_))
        result = self.ise.exec(family='device_administration_authorization_exception_rules', function='update_device_admin_local_exception_rule_by_id', params=self.new_object).response
        return result

    def delete(self):
        id = self.new_object.get('id')
        name = False
        if self.new_object.get('rule', {}) is not None:
            name = self.new_object.get('rule', {}).get('name')
            id = id or self.new_object.get('rule', {}).get('id')
        policy_id = self.new_object.get('policy_id')
        result = None
        if not id:
            id_ = self.get_object_by_name(name, policy_id).get('rule', {}).get('id')
            rule = self.new_object.get('rule', {})
            rule.update(dict(id=id_))
            self.new_object.update(dict(rule=rule, id=id_))
        result = self.ise.exec(family='device_administration_authorization_exception_rules', function='delete_device_admin_local_exception_rule_by_id', params=self.new_object).response
        return result