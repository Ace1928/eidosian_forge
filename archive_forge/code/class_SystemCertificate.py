from __future__ import absolute_import, division, print_function
from ansible.plugins.action import ActionBase
from ansible.errors import AnsibleActionFail
from ansible_collections.cisco.ise.plugins.plugin_utils.ise import (
class SystemCertificate(object):

    def __init__(self, params, ise):
        self.ise = ise
        self.new_object = dict(admin=params.get('admin'), allow_portal_tag_transfer_for_same_subject=params.get('allowPortalTagTransferForSameSubject'), allow_replacement_of_portal_group_tag=params.get('allowReplacementOfPortalGroupTag'), allow_role_transfer_for_same_subject=params.get('allowRoleTransferForSameSubject'), description=params.get('description'), eap=params.get('eap'), expiration_ttl_period=params.get('expirationTTLPeriod'), expiration_ttl_units=params.get('expirationTTLUnits'), ims=params.get('ims'), name=params.get('name'), portal=params.get('portal'), portal_group_tag=params.get('portalGroupTag'), pxgrid=params.get('pxgrid'), radius=params.get('radius'), renew_self_signed_certificate=params.get('renewSelfSignedCertificate'), saml=params.get('saml'), id=params.get('id'), host_name=params.get('hostName'), allow_wildcard_delete=params.get('allowWildcardDelete'))

    def get_object_by_name(self, name, host_name):
        result = None
        gen_items_responses = self.ise.exec(family='certificates', function='get_system_certificates_generator', params={'host_name': host_name})
        try:
            for items_response in gen_items_responses:
                items = items_response.response.get('response', []) or []
                result = get_dict_result(items, 'friendlyName', name)
                if result:
                    return result
        except (TypeError, AttributeError) as e:
            self.ise.fail_json(msg='An error occured when executing operation. Check the configuration of your API Settings and API Gateway settings on your ISE server. This collection assumes that the API Gateway, the ERS APIs and OpenAPIs are enabled. You may want to enable the (ise_debug: True) argument. The error was: {error}'.format(error=e))
        except Exception:
            result = None
        return result

    def get_object_by_id(self, id, host_name):
        try:
            result = self.ise.exec(family='certificates', function='get_system_certificate_by_id', params={'id': id, 'host_name': host_name}, handle_func_exception=False).response['response']
        except Exception as e:
            result = None
        return result

    def exists(self):
        prev_obj = None
        result = False
        id = self.new_object.get('id')
        name = self.new_object.get('name')
        host_name = self.new_object.get('host_name')
        if id:
            prev_obj = self.get_object_by_id(id, host_name)
            result = prev_obj is not None and isinstance(prev_obj, dict)
        elif name:
            prev_obj = self.get_object_by_name(name, host_name)
            result = prev_obj is not None and isinstance(prev_obj, dict)
        return (result, prev_obj)

    def requires_update(self, current_obj):
        requested_obj = self.new_object
        used_by_value = current_obj.get('usedBy')
        if used_by_value is None or used_by_value.lower() == 'not in use':
            current_obj['eap'] = False
            current_obj['pxgrid'] = False
            current_obj['radius'] = False
            current_obj['ims'] = False
        else:
            current_obj['eap'] = 'eap' in used_by_value.lower()
            current_obj['pxgrid'] = 'pxgrid' in used_by_value.lower()
            current_obj['radius'] = 'radius' in used_by_value.lower()
            current_obj['ims'] = 'ims' in used_by_value.lower()
        obj_params = [('admin', 'admin'), ('allowPortalTagTransferForSameSubject', 'allow_portal_tag_transfer_for_same_subject'), ('allowReplacementOfPortalGroupTag', 'allow_replacement_of_portal_group_tag'), ('allowRoleTransferForSameSubject', 'allow_role_transfer_for_same_subject'), ('description', 'description'), ('eap', 'eap'), ('expirationTTLPeriod', 'expiration_ttl_period'), ('expirationTTLUnits', 'expiration_ttl_units'), ('ims', 'ims'), ('friendlyName', 'name'), ('portal', 'portal'), ('portalGroupTag', 'portal_group_tag'), ('pxgrid', 'pxgrid'), ('radius', 'radius'), ('renewSelfSignedCertificate', 'renew_self_signed_certificate'), ('saml', 'saml'), ('id', 'id'), ('hostName', 'host_name'), ('allowWildcardDelete', 'allow_wildcard_delete')]
        return any((not ise_compare_equality(current_obj.get(ise_param), requested_obj.get(ansible_param)) for ise_param, ansible_param in obj_params))

    def update(self):
        id = self.new_object.get('id')
        name = self.new_object.get('name')
        host_name = self.new_object.get('host_name')
        result = None
        if not id:
            id_ = self.get_object_by_name(name, host_name).get('id')
            self.new_object.update(dict(id=id_))
        result = self.ise.exec(family='certificates', function='update_system_certificate', params=self.new_object).response
        return result

    def delete(self):
        id = self.new_object.get('id')
        name = self.new_object.get('name')
        host_name = self.new_object.get('host_name')
        result = None
        if not id:
            id_ = self.get_object_by_name(name, host_name).get('id')
            self.new_object.update(dict(id=id_))
        result = self.ise.exec(family='certificates', function='delete_system_certificate_by_id', params=self.new_object).response
        return result