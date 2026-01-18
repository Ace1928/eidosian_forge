from __future__ import (absolute_import, division, print_function)
from ansible.plugins.action import ActionBase
from ansible.errors import AnsibleActionFail
from ansible_collections.cisco.dnac.plugins.plugin_utils.dnac import (
from ansible_collections.cisco.dnac.plugins.plugin_utils.exceptions import (
class EventEmailConfig(object):

    def __init__(self, params, dnac):
        self.dnac = dnac
        self.new_object = dict(emailConfigId=params.get('emailConfigId'), primarySMTPConfig=params.get('primarySMTPConfig'), secondarySMTPConfig=params.get('secondarySMTPConfig'), fromEmail=params.get('fromEmail'), toEmail=params.get('toEmail'), subject=params.get('subject'))

    def get_all_params(self, name=None, id=None):
        new_object_params = {}
        return new_object_params

    def create_params(self):
        new_object_params = {}
        new_object_params['emailConfigId'] = self.new_object.get('emailConfigId')
        new_object_params['primarySMTPConfig'] = self.new_object.get('primarySMTPConfig')
        new_object_params['secondarySMTPConfig'] = self.new_object.get('secondarySMTPConfig')
        new_object_params['fromEmail'] = self.new_object.get('fromEmail')
        new_object_params['toEmail'] = self.new_object.get('toEmail')
        new_object_params['subject'] = self.new_object.get('subject')
        return new_object_params

    def update_all_params(self):
        new_object_params = {}
        new_object_params['emailConfigId'] = self.new_object.get('emailConfigId')
        new_object_params['primarySMTPConfig'] = self.new_object.get('primarySMTPConfig')
        new_object_params['secondarySMTPConfig'] = self.new_object.get('secondarySMTPConfig')
        new_object_params['fromEmail'] = self.new_object.get('fromEmail')
        new_object_params['toEmail'] = self.new_object.get('toEmail')
        new_object_params['subject'] = self.new_object.get('subject')
        return new_object_params

    def get_object_by_name(self, name):
        result = None
        try:
            items = self.dnac.exec(family='event_management', function='get_email_destination', params=self.get_all_params(name=name))
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
        requested_obj = self.new_object
        obj_params = [('emailConfigId', 'emailConfigId'), ('primarySMTPConfig', 'primarySMTPConfig'), ('secondarySMTPConfig', 'secondarySMTPConfig'), ('fromEmail', 'fromEmail'), ('toEmail', 'toEmail'), ('subject', 'subject')]
        return any((not dnac_compare_equality(current_obj.get(dnac_param), requested_obj.get(ansible_param)) for dnac_param, ansible_param in obj_params))

    def create(self):
        result = self.dnac.exec(family='event_management', function='create_email_destination', params=self.create_params(), op_modifies=True)
        return result

    def update(self):
        id = self.new_object.get('id')
        name = self.new_object.get('name')
        result = None
        result = self.dnac.exec(family='event_management', function='update_email_destination', params=self.update_all_params(), op_modifies=True)
        return result