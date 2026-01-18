from __future__ import absolute_import, division, print_function
from ansible.plugins.action import ActionBase
from ansible.errors import AnsibleActionFail
from ansible_collections.cisco.ise.plugins.plugin_utils.ise import (
class GuestUser(object):

    def __init__(self, params, ise):
        self.ise = ise
        self.new_object = dict(name=params.get('name'), description=params.get('description'), guest_type=params.get('guestType'), status=params.get('status'), status_reason=params.get('statusReason'), reason_for_visit=params.get('reasonForVisit'), sponsor_user_id=params.get('sponsorUserId'), sponsor_user_name=params.get('sponsorUserName'), guest_info=params.get('guestInfo'), guest_access_info=params.get('guestAccessInfo'), portal_id=params.get('portalId'), custom_fields=params.get('customFields'), id=params.get('id'))

    def get_object_by_name(self, name):
        try:
            result = self.ise.exec(family='guest_user', function='get_guest_user_by_name', params={'name': name}, handle_func_exception=False).response['GuestUser']
            result = get_dict_result(result, 'name', name)
        except (TypeError, AttributeError) as e:
            self.ise.fail_json(msg='An error occured when executing operation. Check the configuration of your API Settings and API Gateway settings on your ISE server. This collection assumes that the API Gateway, the ERS APIs and OpenAPIs are enabled. You may want to enable the (ise_debug: True) argument. The error was: {error}'.format(error=e))
        except Exception:
            result = None
        return result

    def get_object_by_id(self, id):
        try:
            result = self.ise.exec(family='guest_user', function='get_guest_user_by_id', handle_func_exception=False, params={'id': id}).response['GuestUser']
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
        obj_params = [('name', 'name'), ('description', 'description'), ('guestType', 'guest_type'), ('status', 'status'), ('statusReason', 'status_reason'), ('reasonForVisit', 'reason_for_visit'), ('sponsorUserId', 'sponsor_user_id'), ('sponsorUserName', 'sponsor_user_name'), ('guestInfo', 'guest_info'), ('guestAccessInfo', 'guest_access_info'), ('portalId', 'portal_id'), ('customFields', 'custom_fields'), ('id', 'id')]
        return any((not ise_compare_equality(current_obj.get(ise_param), requested_obj.get(ansible_param)) for ise_param, ansible_param in obj_params))

    def create(self):
        result = self.ise.exec(family='guest_user', function='create_guest_user', params=self.new_object).response
        return result

    def update(self):
        id = self.new_object.get('id')
        name = self.new_object.get('name')
        result = None
        if id:
            result = self.ise.exec(family='guest_user', function='update_guest_user_by_id', params=self.new_object).response
        elif name:
            result = self.ise.exec(family='guest_user', function='update_guest_user_by_name', params=self.new_object).response
        return result

    def delete(self):
        id = self.new_object.get('id')
        name = self.new_object.get('name')
        result = None
        if id:
            result = self.ise.exec(family='guest_user', function='delete_guest_user_by_id', params=self.new_object).response
        elif name:
            result = self.ise.exec(family='guest_user', function='delete_guest_user_by_name', params=self.new_object).response
        return result