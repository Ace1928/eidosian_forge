from __future__ import absolute_import, division, print_function
from ansible.plugins.action import ActionBase
from ansible.errors import AnsibleActionFail
from ansible_collections.cisco.ise.plugins.plugin_utils.ise import (
class AllowedProtocols(object):

    def __init__(self, params, ise):
        self.ise = ise
        self.new_object = dict(name=params.get('name'), description=params.get('description'), eap_tls=params.get('eapTls'), peap=params.get('peap'), eap_fast=params.get('eapFast'), eap_ttls=params.get('eapTtls'), teap=params.get('teap'), process_host_lookup=params.get('processHostLookup'), allow_pap_ascii=params.get('allowPapAscii'), allow_chap=params.get('allowChap'), allow_ms_chap_v1=params.get('allowMsChapV1'), allow_ms_chap_v2=params.get('allowMsChapV2'), allow_eap_md5=params.get('allowEapMd5'), allow_leap=params.get('allowLeap'), allow_eap_tls=params.get('allowEapTls'), allow_eap_ttls=params.get('allowEapTtls'), allow_eap_fast=params.get('allowEapFast'), allow_peap=params.get('allowPeap'), allow_teap=params.get('allowTeap'), allow_preferred_eap_protocol=params.get('allowPreferredEapProtocol'), preferred_eap_protocol=params.get('preferredEapProtocol'), eap_tls_l_bit=params.get('eapTlsLBit'), allow_weak_ciphers_for_eap=params.get('allowWeakCiphersForEap'), require_message_auth=params.get('requireMessageAuth'), id=params.get('id'))

    def get_object_by_name(self, name):
        try:
            result = self.ise.exec(family='allowed_protocols', function='get_allowed_protocol_by_name', params={'name': name}, handle_func_exception=False).response['AllowedProtocols']
            result = get_dict_result(result, 'name', name)
        except (TypeError, AttributeError) as e:
            self.ise.fail_json(msg='An error occured when executing operation. Check the configuration of your API Settings and API Gateway settings on your ISE server. This collection assumes that the API Gateway, the ERS APIs and OpenAPIs are enabled. You may want to enable the (ise_debug: True) argument. The error was: {error}'.format(error=e))
        except Exception:
            result = None
        return result

    def get_object_by_id(self, id):
        try:
            result = self.ise.exec(family='allowed_protocols', function='get_allowed_protocol_by_id', handle_func_exception=False, params={'id': id}).response['AllowedProtocols']
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
        obj_params = [('name', 'name'), ('description', 'description'), ('eapTls', 'eap_tls'), ('peap', 'peap'), ('eapFast', 'eap_fast'), ('eapTtls', 'eap_ttls'), ('teap', 'teap'), ('processHostLookup', 'process_host_lookup'), ('allowPapAscii', 'allow_pap_ascii'), ('allowChap', 'allow_chap'), ('allowMsChapV1', 'allow_ms_chap_v1'), ('allowMsChapV2', 'allow_ms_chap_v2'), ('allowEapMd5', 'allow_eap_md5'), ('allowLeap', 'allow_leap'), ('allowEapTls', 'allow_eap_tls'), ('allowEapTtls', 'allow_eap_ttls'), ('allowEapFast', 'allow_eap_fast'), ('allowPeap', 'allow_peap'), ('allowTeap', 'allow_teap'), ('allowPreferredEapProtocol', 'allow_preferred_eap_protocol'), ('preferredEapProtocol', 'preferred_eap_protocol'), ('eapTlsLBit', 'eap_tls_l_bit'), ('allowWeakCiphersForEap', 'allow_weak_ciphers_for_eap'), ('requireMessageAuth', 'require_message_auth'), ('id', 'id')]
        return any((not ise_compare_equality(current_obj.get(ise_param), requested_obj.get(ansible_param)) for ise_param, ansible_param in obj_params))

    def create(self):
        result = self.ise.exec(family='allowed_protocols', function='create_allowed_protocol', params=self.new_object).response
        return result

    def update(self):
        id = self.new_object.get('id')
        name = self.new_object.get('name')
        result = None
        if not id:
            id_ = self.get_object_by_name(name).get('id')
            self.new_object.update(dict(id=id_))
        result = self.ise.exec(family='allowed_protocols', function='update_allowed_protocol_by_id', params=self.new_object).response
        return result

    def delete(self):
        id = self.new_object.get('id')
        name = self.new_object.get('name')
        result = None
        if not id:
            id_ = self.get_object_by_name(name).get('id')
            self.new_object.update(dict(id=id_))
        result = self.ise.exec(family='allowed_protocols', function='delete_allowed_protocol_by_id', params=self.new_object).response
        return result