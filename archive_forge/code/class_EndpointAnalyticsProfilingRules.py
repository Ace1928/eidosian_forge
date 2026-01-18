from __future__ import (absolute_import, division, print_function)
from ansible.plugins.action import ActionBase, Display
from ansible.errors import AnsibleActionFail
from ansible_collections.cisco.dnac.plugins.plugin_utils.dnac import (
from ansible_collections.cisco.dnac.plugins.plugin_utils.exceptions import (
class EndpointAnalyticsProfilingRules(object):

    def __init__(self, params, dnac):
        self.dnac = dnac
        self.new_object = dict(ruleId=params.get('ruleId'), ruleName=params.get('ruleName'), ruleType=params.get('ruleType'), ruleVersion=params.get('ruleVersion'), rulePriority=params.get('rulePriority'), sourcePriority=params.get('sourcePriority'), isDeleted=params.get('isDeleted'), lastModifiedBy=params.get('lastModifiedBy'), lastModifiedOn=params.get('lastModifiedOn'), pluginId=params.get('pluginId'), clusterId=params.get('clusterId'), rejected=params.get('rejected'), result=params.get('result'), conditionGroups=params.get('conditionGroups'), usedAttributes=params.get('usedAttributes'), rule_id=params.get('ruleId'))

    def get_all_params(self, name=None, id=None):
        new_object_params = {}
        new_object_params['rule_type'] = self.new_object.get('ruleType')
        return new_object_params

    def create_params(self):
        new_object_params = {}
        new_object_params['ruleId'] = self.new_object.get('ruleId')
        new_object_params['ruleName'] = self.new_object.get('ruleName')
        new_object_params['ruleType'] = self.new_object.get('ruleType')
        new_object_params['ruleVersion'] = self.new_object.get('ruleVersion')
        new_object_params['rulePriority'] = self.new_object.get('rulePriority')
        new_object_params['sourcePriority'] = self.new_object.get('sourcePriority')
        new_object_params['isDeleted'] = self.new_object.get('isDeleted')
        new_object_params['lastModifiedBy'] = self.new_object.get('lastModifiedBy')
        new_object_params['lastModifiedOn'] = self.new_object.get('lastModifiedOn')
        new_object_params['pluginId'] = self.new_object.get('pluginId')
        new_object_params['clusterId'] = self.new_object.get('clusterId')
        new_object_params['rejected'] = self.new_object.get('rejected')
        new_object_params['result'] = self.new_object.get('result')
        new_object_params['conditionGroups'] = self.new_object.get('conditionGroups')
        new_object_params['usedAttributes'] = self.new_object.get('usedAttributes')
        return new_object_params

    def delete_by_id_params(self):
        new_object_params = {}
        new_object_params['rule_id'] = self.new_object.get('rule_id')
        return new_object_params

    def update_by_id_params(self):
        new_object_params = {}
        new_object_params['ruleId'] = self.new_object.get('ruleId')
        new_object_params['ruleName'] = self.new_object.get('ruleName')
        new_object_params['ruleType'] = self.new_object.get('ruleType')
        new_object_params['ruleVersion'] = self.new_object.get('ruleVersion')
        new_object_params['rulePriority'] = self.new_object.get('rulePriority')
        new_object_params['sourcePriority'] = self.new_object.get('sourcePriority')
        new_object_params['isDeleted'] = self.new_object.get('isDeleted')
        new_object_params['lastModifiedBy'] = self.new_object.get('lastModifiedBy')
        new_object_params['lastModifiedOn'] = self.new_object.get('lastModifiedOn')
        new_object_params['pluginId'] = self.new_object.get('pluginId')
        new_object_params['clusterId'] = self.new_object.get('clusterId')
        new_object_params['rejected'] = self.new_object.get('rejected')
        new_object_params['result'] = self.new_object.get('result')
        new_object_params['conditionGroups'] = self.new_object.get('conditionGroups')
        new_object_params['usedAttributes'] = self.new_object.get('usedAttributes')
        return new_object_params

    def get_object_by_name(self, name):
        result = None
        try:
            items = self.dnac.exec(family='policy', function='get_list_of_profiling_rules', params=self.get_all_params(name=name))
            if isinstance(items, dict):
                items = items['profilingRules']
            result = get_dict_result(items, 'name', name)
        except Exception:
            result = None
        return result

    def get_object_by_id(self, id):
        result = None
        try:
            items = self.dnac.exec(family='policy', function='get_details_of_a_single_profiling_rule', params={'rule_id': id})
            if isinstance(items, dict):
                if 'response' in items:
                    items = items.get('response')
            result = get_dict_result(items, 'ruleId', id)
        except Exception:
            result = None
        return result

    def exists(self):
        id_exists = False
        name_exists = False
        prev_obj = None
        o_id = self.new_object.get('id')
        o_id = o_id or self.new_object.get('rule_id')
        name = self.new_object.get('name')
        if o_id:
            prev_obj = self.get_object_by_id(o_id)
            id_exists = prev_obj is not None and isinstance(prev_obj, dict)
        if not id_exists and name:
            prev_obj = self.get_object_by_name(name)
            name_exists = prev_obj is not None and isinstance(prev_obj, dict)
        if name_exists:
            _id = prev_obj.get('id')
            _id = _id or prev_obj.get('ruleId')
            if id_exists and name_exists and (o_id != _id):
                raise InconsistentParameters("The 'id' and 'name' params don't refer to the same object")
            if _id:
                self.new_object.update(dict(id=_id))
                self.new_object.update(dict(rule_id=_id))
            if _id:
                prev_obj = self.get_object_by_id(_id)
        it_exists = prev_obj is not None and isinstance(prev_obj, dict)
        return (it_exists, prev_obj)

    def requires_update(self, current_obj):
        requested_obj = self.new_object
        obj_params = [('ruleId', 'ruleId'), ('ruleName', 'ruleName'), ('ruleType', 'ruleType'), ('ruleVersion', 'ruleVersion'), ('rulePriority', 'rulePriority'), ('sourcePriority', 'sourcePriority'), ('isDeleted', 'isDeleted'), ('lastModifiedBy', 'lastModifiedBy'), ('lastModifiedOn', 'lastModifiedOn'), ('pluginId', 'pluginId'), ('clusterId', 'clusterId'), ('rejected', 'rejected'), ('result', 'result'), ('conditionGroups', 'conditionGroups'), ('usedAttributes', 'usedAttributes'), ('ruleId', 'rule_id')]
        return any((not dnac_compare_equality(current_obj.get(dnac_param), requested_obj.get(ansible_param)) for dnac_param, ansible_param in obj_params))

    def create(self):
        result = self.dnac.exec(family='policy', function='create_a_profiling_rule', params=self.create_params(), op_modifies=True)
        return result

    def update(self):
        id = self.new_object.get('id')
        id = id or self.new_object.get('rule_id')
        name = self.new_object.get('name')
        result = None
        if not id:
            prev_obj_name = self.get_object_by_name(name)
            id_ = None
            if prev_obj_name:
                id_ = prev_obj_name.get('id')
                id_ = id_ or prev_obj_name.get('ruleId')
            if id_:
                self.new_object.update(dict(rule_id=id_))
        result = self.dnac.exec(family='policy', function='update_an_existing_profiling_rule', params=self.update_by_id_params(), op_modifies=True)
        return result

    def delete(self):
        id = self.new_object.get('id')
        id = id or self.new_object.get('rule_id')
        name = self.new_object.get('name')
        result = None
        if not id:
            prev_obj_name = self.get_object_by_name(name)
            id_ = None
            if prev_obj_name:
                id_ = prev_obj_name.get('id')
                id_ = id_ or prev_obj_name.get('ruleId')
            if id_:
                self.new_object.update(dict(rule_id=id_))
        result = self.dnac.exec(family='policy', function='delete_an_existing_profiling_rule', params=self.delete_by_id_params())
        return result