from __future__ import absolute_import, division, print_function
import datetime
import math
import re
import time
import traceback
from collections import namedtuple
from ansible.module_utils.basic import (
from ansible.module_utils.parsing.convert_bool import BOOLEANS_TRUE
from ansible.module_utils.six import (
from ansible.module_utils.urls import urlparse
from ipaddress import ip_interface
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.urls import parseStats
from ..module_utils.icontrol import (
from ..module_utils.ipaddress import is_valid_ip
from ..module_utils.teem import send_teem
class AsmPolicyFactParameters(BaseParameters):
    api_map = {'hasParent': 'has_parent', 'protocolIndependent': 'protocol_independent', 'virtualServers': 'virtual_servers', 'manualVirtualServers': 'manual_virtual_servers', 'allowedResponseCodes': 'allowed_response_codes', 'learningMode': 'learning_mode', 'enforcementMode': 'enforcement_mode', 'customXffHeaders': 'custom_xff_headers', 'caseInsensitive': 'case_insensitive', 'stagingSettings': 'staging_settings', 'applicationLanguage': 'application_language', 'trustXff': 'trust_xff', 'geolocation-enforcement': 'geolocation_enforcement', 'disallowedLocations': 'disallowed_locations', 'signature-settings': 'signature_settings', 'header-settings': 'header_settings', 'cookie-settings': 'cookie_settings', 'policy-builder': 'policy_builder', 'disallowed-geolocations': 'disallowed_geolocations', 'whitelist-ips': 'whitelist_ips', 'fullPath': 'full_path', 'csrf-protection': 'csrf_protection', 'isModified': 'apply'}
    returnables = ['full_path', 'name', 'policy_id', 'active', 'protocol_independent', 'has_parent', 'type', 'virtual_servers', 'allowed_response_codes', 'description', 'learning_mode', 'enforcement_mode', 'custom_xff_headers', 'case_insensitive', 'signature_staging', 'place_signatures_in_staging', 'enforcement_readiness_period', 'path_parameter_handling', 'trigger_asm_irule_event', 'inspect_http_uploads', 'mask_credit_card_numbers_in_request', 'maximum_http_header_length', 'use_dynamic_session_id_in_url', 'maximum_cookie_header_length', 'application_language', 'trust_xff', 'disallowed_geolocations', 'csrf_urls', 'csrf_protection_enabled', 'csrf_protection_ssl_only', 'csrf_protection_expiration_time_in_seconds', 'apply']

    def _morph_keys(self, key_map, item):
        for k, v in iteritems(key_map):
            item[v] = item.pop(k, None)
        result = self._filter_params(item)
        return result

    @property
    def active(self):
        return flatten_boolean(self._values['active'])

    @property
    def apply(self):
        return flatten_boolean(self._values['apply'])

    @property
    def case_insensitive(self):
        return flatten_boolean(self._values['case_insensitive'])

    @property
    def has_parent(self):
        return flatten_boolean(self._values['has_parent'])

    @property
    def policy_id(self):
        if self._values['id'] is None:
            return None
        return self._values['id']

    @property
    def manual_virtual_servers(self):
        if 'manual_virtual_servers' in self._values:
            if self._values['manual_virtual_servers'] is None:
                return None
            return self._values['manual_virtual_servers']

    @property
    def signature_staging(self):
        if 'staging_settings' in self._values:
            if self._values['staging_settings'] is None:
                return None
            if 'signatureStaging' in self._values['staging_settings']:
                return flatten_boolean(self._values['staging_settings']['signatureStaging'])
        if 'signature_settings' in self._values:
            if self._values['signature_settings'] is None:
                return None
            if 'signatureStaging' in self._values['signature_settings']:
                return flatten_boolean(self._values['signature_settings']['signatureStaging'])

    @property
    def place_signatures_in_staging(self):
        if 'staging_settings' in self._values:
            if self._values['staging_settings'] is None:
                return None
            if 'placeSignaturesInStaging' in self._values['staging_settings']:
                return flatten_boolean(self._values['staging_settings']['placeSignaturesInStaging'])
        if 'signature_settings' in self._values:
            if self._values['signature_settings'] is None:
                return None
            if 'signatureStaging' in self._values['signature_settings']:
                return flatten_boolean(self._values['signature_settings']['placeSignaturesInStaging'])

    @property
    def enforcement_readiness_period(self):
        if 'staging_settings' in self._values:
            if self._values['staging_settings'] is None:
                return None
            if 'enforcementReadinessPeriod' in self._values['staging_settings']:
                return self._values['staging_settings']['enforcementReadinessPeriod']
        if 'general' in self._values:
            if self._values['general'] is None:
                return None
            if 'signatureStaging' in self._values['general']:
                return self._values['general']['enforcementReadinessPeriod']

    @property
    def path_parameter_handling(self):
        if 'attributes' in self._values:
            if self._values['attributes'] is None:
                return None
            if 'pathParameterHandling' in self._values['attributes']:
                return self._values['attributes']['pathParameterHandling']
        if 'general' in self._values:
            if self._values['general'] is None:
                return None
            if 'pathParameterHandling' in self._values['general']:
                return self._values['general']['pathParameterHandling']

    @property
    def trigger_asm_irule_event(self):
        if 'attributes' in self._values:
            if self._values['attributes'] is None:
                return None
            if 'triggerAsmIruleEvent' in self._values['attributes']:
                return self._values['attributes']['triggerAsmIruleEvent']
        if 'general' in self._values:
            if self._values['general'] is None:
                return None
            if 'triggerAsmIruleEvent' in self._values['general']:
                return self._values['general']['triggerAsmIruleEvent']

    @property
    def inspect_http_uploads(self):
        if 'attributes' in self._values:
            if self._values['attributes'] is None:
                return None
            if 'inspectHttpUploads' in self._values['attributes']:
                return flatten_boolean(self._values['attributes']['inspectHttpUploads'])
        if 'antivirus' in self._values:
            if self._values['antivirus'] is None:
                return None
            if 'inspectHttpUploads' in self._values['antivirus']:
                return flatten_boolean(self._values['antivirus']['inspectHttpUploads'])

    @property
    def mask_credit_card_numbers_in_request(self):
        if 'attributes' in self._values:
            if self._values['attributes'] is None:
                return None
            if 'maskCreditCardNumbersInRequest' in self._values['attributes']:
                return flatten_boolean(self._values['attributes']['maskCreditCardNumbersInRequest'])
        if 'general' in self._values:
            if self._values['general'] is None:
                return None
            if 'maskCreditCardNumbersInRequest' in self._values['general']:
                return flatten_boolean(self._values['general']['maskCreditCardNumbersInRequest'])

    @property
    def maximum_http_header_length(self):
        if 'attributes' in self._values:
            if self._values['attributes'] is None:
                return None
            if 'maximumHttpHeaderLength' in self._values['attributes']:
                if self._values['attributes']['maximumHttpHeaderLength'] == 'any':
                    return 'any'
                return int(self._values['attributes']['maximumHttpHeaderLength'])
        if 'header_settings' in self._values:
            if self._values['header_settings'] is None:
                return None
            if 'maximumHttpHeaderLength' in self._values['header_settings']:
                if self._values['header_settings']['maximumHttpHeaderLength'] == 'any':
                    return 'any'
                return int(self._values['header_settings']['maximumHttpHeaderLength'])

    @property
    def use_dynamic_session_id_in_url(self):
        if 'attributes' in self._values:
            if self._values['attributes'] is None:
                return None
            if 'useDynamicSessionIdInUrl' in self._values['attributes']:
                return flatten_boolean(self._values['attributes']['useDynamicSessionIdInUrl'])
        if 'general' in self._values:
            if self._values['general'] is None:
                return None
            if 'useDynamicSessionIdInUrl' in self._values['general']:
                return flatten_boolean(self._values['general']['useDynamicSessionIdInUrl'])

    @property
    def maximum_cookie_header_length(self):
        if 'attributes' in self._values:
            if self._values['attributes'] is None:
                return None
            if 'maximumCookieHeaderLength' in self._values['attributes']:
                if self._values['attributes']['maximumCookieHeaderLength'] == 'any':
                    return 'any'
                return int(self._values['attributes']['maximumCookieHeaderLength'])
        if 'cookie_settings' in self._values:
            if self._values['cookie_settings'] is None:
                return None
            if 'maximumCookieHeaderLength' in self._values['cookie_settings']:
                if self._values['cookie_settings']['maximumCookieHeaderLength'] == 'any':
                    return 'any'
                return int(self._values['cookie_settings']['maximumCookieHeaderLength'])

    @property
    def trust_xff(self):
        if 'trust_xff' in self._values:
            if self._values['trust_xff'] is None:
                return None
            return flatten_boolean(self._values['trust_xff'])
        if 'general' in self._values:
            if self._values['general'] is None:
                return None
            if 'trustXff' in self._values['general']:
                return flatten_boolean(self._values['general']['trustXff'])

    @property
    def custom_xff_headers(self):
        if 'custom_xff_headers' in self._values:
            if self._values['custom_xff_headers'] is None:
                return None
            return self._values['custom_xff_headers']
        if 'general' in self._values:
            if self._values['general'] is None:
                return None
            if 'customXffHeaders' in self._values['general']:
                return self._values['general']['customXffHeaders']

    @property
    def allowed_response_codes(self):
        if 'allowed_response_codes' in self._values:
            if self._values['allowed_response_codes'] is None:
                return None
            return self._values['allowed_response_codes']
        if 'general' in self._values:
            if self._values['general'] is None:
                return None
            if 'allowedResponseCodes' in self._values['general']:
                return self._values['general']['allowedResponseCodes']

    @property
    def learning_mode(self):
        if 'policy_builder' in self._values:
            if self._values['policy_builder'] is None:
                return None
            if 'learningMode' in self._values['policy_builder']:
                return self._values['policy_builder']['learningMode']

    @property
    def disallowed_locations(self):
        if 'geolocation_enforcement' in self._values:
            if self._values['geolocation_enforcement'] is None:
                return None
            return self._values['geolocation_enforcement']['disallowedLocations']

    @property
    def disallowed_geolocations(self):
        if 'disallowed_geolocations' in self._values:
            if self._values['disallowed_geolocations'] is None:
                return None
            return self._values['disallowed_geolocations']

    @property
    def csrf_protection_enabled(self):
        if 'csrf_protection' in self._values:
            return flatten_boolean(self._values['csrf_protection']['enabled'])

    @property
    def csrf_protection_ssl_only(self):
        if 'csrf_protection' in self._values:
            if 'sslOnly' in self._values['csrf_protection']:
                return flatten_boolean(self._values['csrf_protection']['sslOnly'])

    @property
    def csrf_protection_expiration_time_in_seconds(self):
        if 'csrf_protection' in self._values:
            if 'expirationTimeInSeconds' in self._values['csrf_protection']:
                if self._values['csrf_protection']['expirationTimeInSeconds'] is None:
                    return None
                if self._values['csrf_protection']['expirationTimeInSeconds'] == 'disabled':
                    return 'disabled'
                return int(self._values['csrf_protection']['expirationTimeInSeconds'])

    def format_csrf_collection(self, items):
        result = list()
        key_map = {'requiredParameters': 'csrf_url_required_parameters', 'url': 'csrf_url', 'method': 'csrf_url_method', 'enforcementAction': 'csrf_url_enforcement_action', 'id': 'csrf_url_id', 'wildcardOrder': 'csrf_url_wildcard_order', 'parametersList': 'csrf_url_parameters_list'}
        for item in items:
            self._remove_internal_keywords(item)
            item.pop('lastUpdateMicros')
            output = self._morph_keys(key_map, item)
            result.append(output)
        return result

    @property
    def csrf_urls(self):
        if 'csrfUrls' in self._values:
            if self._values['csrfUrls'] is None:
                return None
            return self._values['csrfUrls']
        if 'csrf-urls' in self._values:
            if self._values['csrf-urls'] is None:
                return None
            return self.format_csrf_collection(self._values['csrf-urls'])

    @property
    def protocol_independent(self):
        return flatten_boolean(self._values['protocol_independent'])