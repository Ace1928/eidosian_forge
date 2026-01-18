from __future__ import absolute_import, division, print_function
from ansible.plugins.action import ActionBase
from ansible.errors import AnsibleActionFail
from ansible_collections.cisco.ise.plugins.plugin_utils.ise import (
class AuthorizationProfile(object):

    def __init__(self, params, ise):
        self.ise = ise
        self.new_object = dict(id=params.get('id'), name=params.get('name'), description=params.get('description'), advanced_attributes=params.get('advancedAttributes'), access_type=params.get('accessType'), authz_profile_type=params.get('authzProfileType'), vlan=params.get('vlan'), reauth=params.get('reauth'), airespace_acl=params.get('airespaceACL'), airespace_ipv6_acl=params.get('airespaceIPv6ACL'), web_redirection=params.get('webRedirection'), acl=params.get('acl'), track_movement=params.get('trackMovement'), agentless_posture=params.get('agentlessPosture'), service_template=params.get('serviceTemplate'), easywired_session_candidate=params.get('easywiredSessionCandidate'), dacl_name=params.get('daclName'), voice_domain_permission=params.get('voiceDomainPermission'), neat=params.get('neat'), web_auth=params.get('webAuth'), auto_smart_port=params.get('autoSmartPort'), interface_template=params.get('interfaceTemplate'), ipv6_acl_filter=params.get('ipv6ACLFilter'), avc_profile=params.get('avcProfile'), mac_sec_policy=params.get('macSecPolicy'), asa_vpn=params.get('asaVpn'), profile_name=params.get('profileName'), ipv6_dacl_name=params.get('ipv6DaclName'))

    def get_object_by_name(self, name):
        try:
            result = self.ise.exec(family='authorization_profile', function='get_authorization_profile_by_name', params={'name': name}, handle_func_exception=False).response['AuthorizationProfile']
            result = get_dict_result(result, 'name', name)
        except (TypeError, AttributeError) as e:
            self.ise.fail_json(msg='An error occured when executing operation. Check the configuration of your API Settings and API Gateway settings on your ISE server. This collection assumes that the API Gateway, the ERS APIs and OpenAPIs are enabled. You may want to enable the (ise_debug: True) argument. The error was: {error}'.format(error=e))
        except Exception:
            result = None
        return result

    def get_object_by_id(self, id):
        try:
            result = self.ise.exec(family='authorization_profile', function='get_authorization_profile_by_id', handle_func_exception=False, params={'id': id}).response['AuthorizationProfile']
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
        obj_params = [('id', 'id'), ('name', 'name'), ('description', 'description'), ('advancedAttributes', 'advanced_attributes'), ('accessType', 'access_type'), ('authzProfileType', 'authz_profile_type'), ('vlan', 'vlan'), ('reauth', 'reauth'), ('airespaceACL', 'airespace_acl'), ('airespaceIPv6ACL', 'airespace_ipv6_acl'), ('webRedirection', 'web_redirection'), ('acl', 'acl'), ('trackMovement', 'track_movement'), ('agentlessPosture', 'agentless_posture'), ('serviceTemplate', 'service_template'), ('easywiredSessionCandidate', 'easywired_session_candidate'), ('daclName', 'dacl_name'), ('voiceDomainPermission', 'voice_domain_permission'), ('neat', 'neat'), ('webAuth', 'web_auth'), ('autoSmartPort', 'auto_smart_port'), ('interfaceTemplate', 'interface_template'), ('ipv6ACLFilter', 'ipv6_acl_filter'), ('avcProfile', 'avc_profile'), ('macSecPolicy', 'mac_sec_policy'), ('asaVpn', 'asa_vpn'), ('profileName', 'profile_name'), ('ipv6DaclName', 'ipv6_dacl_name')]
        return any((not ise_compare_equality(current_obj.get(ise_param), requested_obj.get(ansible_param)) for ise_param, ansible_param in obj_params))

    def create(self):
        result = self.ise.exec(family='authorization_profile', function='create_authorization_profile', params=self.new_object).response
        return result

    def update(self):
        id = self.new_object.get('id')
        name = self.new_object.get('name')
        result = None
        if not id:
            id_ = self.get_object_by_name(name).get('id')
            self.new_object.update(dict(id=id_))
        result = self.ise.exec(family='authorization_profile', function='update_authorization_profile_by_id', params=self.new_object).response
        return result

    def delete(self):
        id = self.new_object.get('id')
        name = self.new_object.get('name')
        result = None
        if not id:
            id_ = self.get_object_by_name(name).get('id')
            self.new_object.update(dict(id=id_))
        result = self.ise.exec(family='authorization_profile', function='delete_authorization_profile_by_id', params=self.new_object).response
        return result