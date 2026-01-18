from __future__ import absolute_import, division, print_function
from ansible.plugins.action import ActionBase
from ansible.errors import AnsibleActionFail
from ansible_collections.cisco.ise.plugins.plugin_utils.ise import (
class CertificateProfile(object):

    def __init__(self, params, ise):
        self.ise = ise
        self.new_object = dict(id=params.get('id'), name=params.get('name'), description=params.get('description'), external_identity_store_name=params.get('externalIdentityStoreName'), certificate_attribute_name=params.get('certificateAttributeName'), allowed_as_user_name=params.get('allowedAsUserName'), match_mode=params.get('matchMode'), username_from=params.get('usernameFrom'))

    def get_object_by_name(self, name):
        try:
            result = self.ise.exec(family='certificate_profile', function='get_certificate_profile_by_name', params={'name': name}, handle_func_exception=False).response['CertificateProfile']
            result = get_dict_result(result, 'name', name)
        except (TypeError, AttributeError) as e:
            self.ise.fail_json(msg='An error occured when executing operation. Check the configuration of your API Settings and API Gateway settings on your ISE server. This collection assumes that the API Gateway, the ERS APIs and OpenAPIs are enabled. You may want to enable the (ise_debug: True) argument. The error was: {error}'.format(error=e))
        except Exception:
            result = None
        return result

    def get_object_by_id(self, id):
        try:
            result = self.ise.exec(family='certificate_profile', function='get_certificate_profile_by_id', params={'id': id}, handle_func_exception=False).response['CertificateProfile']
        except (TypeError, AttributeError) as e:
            self.ise.fail_json(msg='An error occured when executing operation. Check the configuration of your API Settings and API Gateway settings on your ISE server. This collection assumes that the API Gateway, the ERS APIs and OpenAPIs are enabled. You may want to enable the (ise_debug: True) argument. The error was: {error}'.format(error=e))
        except Exception:
            result = None
        return result

    def exists(self):
        prev_obj = None
        result = False
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
        obj_params = [('id', 'id'), ('name', 'name'), ('description', 'description'), ('externalIdentityStoreName', 'external_identity_store_name'), ('certificateAttributeName', 'certificate_attribute_name'), ('allowedAsUserName', 'allowed_as_user_name'), ('matchMode', 'match_mode'), ('usernameFrom', 'username_from')]
        return any((not ise_compare_equality(current_obj.get(ise_param), requested_obj.get(ansible_param)) for ise_param, ansible_param in obj_params))

    def create(self):
        result = self.ise.exec(family='certificate_profile', function='create_certificate_profile', params=self.new_object).response
        return result

    def update(self):
        id = self.new_object.get('id')
        name = self.new_object.get('name')
        result = None
        if not id:
            id_ = self.get_object_by_name(name).get('id')
            self.new_object.update(dict(id=id_))
        result = self.ise.exec(family='certificate_profile', function='update_certificate_profile_by_id', params=self.new_object).response
        return result