from __future__ import absolute_import, division, print_function
from ansible.plugins.action import ActionBase
from ansible.errors import AnsibleActionFail
from ansible_collections.cisco.ise.plugins.plugin_utils.ise import (
from ansible_collections.cisco.ise.plugins.plugin_utils.exceptions import (
class TrustedCertificate(object):

    def __init__(self, params, ise):
        self.ise = ise
        self.new_object = dict(authenticate_before_crl_received=params.get('authenticateBeforeCRLReceived'), automatic_crl_update=params.get('automaticCRLUpdate'), automatic_crl_update_period=params.get('automaticCRLUpdatePeriod'), automatic_crl_update_units=params.get('automaticCRLUpdateUnits'), crl_distribution_url=params.get('crlDistributionUrl'), crl_download_failure_retries=params.get('crlDownloadFailureRetries'), crl_download_failure_retries_units=params.get('crlDownloadFailureRetriesUnits'), description=params.get('description'), download_crl=params.get('downloadCRL'), enable_ocsp_validation=params.get('enableOCSPValidation'), enable_server_identity_check=params.get('enableServerIdentityCheck'), ignore_crl_expiration=params.get('ignoreCRLExpiration'), name=params.get('name'), non_automatic_crl_update_period=params.get('nonAutomaticCRLUpdatePeriod'), non_automatic_crl_update_units=params.get('nonAutomaticCRLUpdateUnits'), reject_if_no_status_from_ocs_p=params.get('rejectIfNoStatusFromOCSP'), reject_if_unreachable_from_ocs_p=params.get('rejectIfUnreachableFromOCSP'), selected_ocsp_service=params.get('selectedOCSPService'), status=params.get('status'), trust_for_certificate_based_admin_auth=params.get('trustForCertificateBasedAdminAuth'), trust_for_cisco_services_auth=params.get('trustForCiscoServicesAuth'), trust_for_client_auth=params.get('trustForClientAuth'), trust_for_ise_auth=params.get('trustForIseAuth'), id=params.get('id'))

    def get_object_by_name(self, name):
        result = None
        gen_items_responses = self.ise.exec(family='certificates', function='get_trusted_certificates_generator')
        try:
            for items_response in gen_items_responses:
                items = items_response.response.get('response', [])
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
            result = self.ise.exec(family='certificates', function='get_trusted_certificate_by_id', params={'id': id}, handle_func_exception=False).response['response']
        except (TypeError, AttributeError) as e:
            self.ise.fail_json(msg='An error occured when executing operation. Check the configuration of your API Settings and API Gateway settings on your ISE server. This collection assumes that the API Gateway, the ERS APIs and OpenAPIs are enabled. You may want to enable the (ise_debug: True) argument. The error was: {error}'.format(error=e))
        except Exception:
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
                prev_obj = self.get_object_by_id(_id)
        it_exists = prev_obj is not None and isinstance(prev_obj, dict)
        return (it_exists, prev_obj)

    def requires_update(self, current_obj):
        requested_obj = self.new_object
        obj_params = [('authenticateBeforeCRLReceived', 'authenticate_before_crl_received'), ('automaticCRLUpdate', 'automatic_crl_update'), ('automaticCRLUpdatePeriod', 'automatic_crl_update_period'), ('automaticCRLUpdateUnits', 'automatic_crl_update_units'), ('crlDistributionUrl', 'crl_distribution_url'), ('crlDownloadFailureRetries', 'crl_download_failure_retries'), ('crlDownloadFailureRetriesUnits', 'crl_download_failure_retries_units'), ('description', 'description'), ('downloadCRL', 'download_crl'), ('enableOCSPValidation', 'enable_ocsp_validation'), ('enableServerIdentityCheck', 'enable_server_identity_check'), ('ignoreCRLExpiration', 'ignore_crl_expiration'), ('name', 'name'), ('nonAutomaticCRLUpdatePeriod', 'non_automatic_crl_update_period'), ('nonAutomaticCRLUpdateUnits', 'non_automatic_crl_update_units'), ('rejectIfNoStatusFromOCSP', 'reject_if_no_status_from_ocs_p'), ('rejectIfUnreachableFromOCSP', 'reject_if_unreachable_from_ocs_p'), ('selectedOCSPService', 'selected_ocsp_service'), ('status', 'status'), ('trustForCertificateBasedAdminAuth', 'trust_for_certificate_based_admin_auth'), ('trustForCiscoServicesAuth', 'trust_for_cisco_services_auth'), ('trustForClientAuth', 'trust_for_client_auth'), ('trustForIseAuth', 'trust_for_ise_auth'), ('id', 'id')]
        return any((not ise_compare_equality(current_obj.get(ise_param), requested_obj.get(ansible_param)) for ise_param, ansible_param in obj_params))

    def update(self):
        id = self.new_object.get('id')
        name = self.new_object.get('name')
        result = None
        if not id:
            id_ = self.get_object_by_name(name).get('id')
            self.new_object.update(dict(id=id_))
        result = self.ise.exec(family='certificates', function='update_trusted_certificate', params=self.new_object).response
        return result

    def delete(self):
        id = self.new_object.get('id')
        name = self.new_object.get('name')
        result = None
        if not id:
            id_ = self.get_object_by_name(name).get('id')
            self.new_object.update(dict(id=id_))
        result = self.ise.exec(family='certificates', function='delete_trusted_certificate_by_id', params=self.new_object).response
        return result