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
class ClientSslProfilesParameters(BaseParameters):
    api_map = {'fullPath': 'full_path', 'alertTimeout': 'alert_timeout', 'allowNonSsl': 'allow_non_ssl', 'authenticateDepth': 'authenticate_depth', 'authenticate': 'authenticate_frequency', 'caFile': 'ca_file', 'cacheSize': 'cache_size', 'cacheTimeout': 'cache_timeout', 'cert': 'certificate_file', 'key': 'key_file', 'chain': 'chain_file', 'crlFile': 'crl_file', 'defaultsFrom': 'parent', 'modSslMethods': 'modssl_methods', 'peerCertMode': 'peer_certification_mode', 'sniRequire': 'sni_require', 'strictResume': 'strict_resume', 'mode': 'profile_mode_enabled', 'renegotiateMaxRecordDelay': 'renegotiation_maximum_record_delay', 'renegotiatePeriod': 'renegotiation_period', 'serverName': 'server_name', 'sessionTicket': 'session_ticket', 'sniDefault': 'sni_default', 'uncleanShutdown': 'unclean_shutdown', 'retainCertificate': 'retain_certificate', 'secureRenegotiation': 'secure_renegotiation_mode', 'handshakeTimeout': 'handshake_timeout', 'certExtensionIncludes': 'forward_proxy_certificate_extension_include', 'certLifespan': 'forward_proxy_certificate_lifespan', 'certLookupByIpaddrPort': 'forward_proxy_lookup_by_ipaddr_port', 'sslForwardProxy': 'forward_proxy_enabled', 'proxyCaPassphrase': 'forward_proxy_ca_passphrase', 'proxyCaCert': 'forward_proxy_ca_certificate_file', 'proxyCaKey': 'forward_proxy_ca_key_file'}
    returnables = ['full_path', 'name', 'alert_timeout', 'allow_non_ssl', 'authenticate_depth', 'authenticate_frequency', 'ca_file', 'cache_size', 'cache_timeout', 'certificate_file', 'key_file', 'chain_file', 'ciphers', 'crl_file', 'parent', 'description', 'modssl_methods', 'peer_certification_mode', 'sni_require', 'sni_default', 'strict_resume', 'profile_mode_enabled', 'renegotiation_maximum_record_delay', 'renegotiation_period', 'renegotiation', 'server_name', 'session_ticket', 'unclean_shutdown', 'retain_certificate', 'secure_renegotiation_mode', 'handshake_timeout', 'forward_proxy_certificate_extension_include', 'forward_proxy_certificate_lifespan', 'forward_proxy_lookup_by_ipaddr_port', 'forward_proxy_enabled', 'forward_proxy_ca_passphrase', 'forward_proxy_ca_certificate_file', 'forward_proxy_ca_key_file']

    @property
    def alert_timeout(self):
        if self._values['alert_timeout'] is None:
            return None
        if self._values['alert_timeout'] == 'indefinite':
            return 0
        return int(self._values['alert_timeout'])

    @property
    def renegotiation_maximum_record_delay(self):
        if self._values['renegotiation_maximum_record_delay'] is None:
            return None
        if self._values['renegotiation_maximum_record_delay'] == 'indefinite':
            return 0
        return int(self._values['renegotiation_maximum_record_delay'])

    @property
    def renegotiation_period(self):
        if self._values['renegotiation_period'] is None:
            return None
        if self._values['renegotiation_period'] == 'indefinite':
            return 0
        return int(self._values['renegotiation_period'])

    @property
    def handshake_timeout(self):
        if self._values['handshake_timeout'] is None:
            return None
        if self._values['handshake_timeout'] == 'indefinite':
            return 0
        return int(self._values['handshake_timeout'])

    @property
    def allow_non_ssl(self):
        if self._values['allow_non_ssl'] is None:
            return None
        if self._values['allow_non_ssl'] == 'disabled':
            return 'no'
        return 'yes'

    @property
    def forward_proxy_enabled(self):
        if self._values['forward_proxy_enabled'] is None:
            return None
        if self._values['forward_proxy_enabled'] == 'disabled':
            return 'no'
        return 'yes'

    @property
    def renegotiation(self):
        if self._values['renegotiation'] is None:
            return None
        if self._values['renegotiation'] == 'disabled':
            return 'no'
        return 'yes'

    @property
    def forward_proxy_lookup_by_ipaddr_port(self):
        if self._values['forward_proxy_lookup_by_ipaddr_port'] is None:
            return None
        if self._values['forward_proxy_lookup_by_ipaddr_port'] == 'disabled':
            return 'no'
        return 'yes'

    @property
    def unclean_shutdown(self):
        if self._values['unclean_shutdown'] is None:
            return None
        if self._values['unclean_shutdown'] == 'disabled':
            return 'no'
        return 'yes'

    @property
    def session_ticket(self):
        if self._values['session_ticket'] is None:
            return None
        if self._values['session_ticket'] == 'disabled':
            return 'no'
        return 'yes'

    @property
    def retain_certificate(self):
        if self._values['retain_certificate'] is None:
            return None
        if self._values['retain_certificate'] == 'true':
            return 'yes'
        return 'no'

    @property
    def server_name(self):
        if self._values['server_name'] in [None, 'none']:
            return None
        return self._values['server_name']

    @property
    def forward_proxy_ca_certificate_file(self):
        if self._values['forward_proxy_ca_certificate_file'] in [None, 'none']:
            return None
        return self._values['forward_proxy_ca_certificate_file']

    @property
    def forward_proxy_ca_key_file(self):
        if self._values['forward_proxy_ca_key_file'] in [None, 'none']:
            return None
        return self._values['forward_proxy_ca_key_file']

    @property
    def authenticate_frequency(self):
        if self._values['authenticate_frequency'] is None:
            return None
        return self._values['authenticate_frequency']

    @property
    def ca_file(self):
        if self._values['ca_file'] in [None, 'none']:
            return None
        return self._values['ca_file']

    @property
    def certificate_file(self):
        if self._values['certificate_file'] in [None, 'none']:
            return None
        return self._values['certificate_file']

    @property
    def key_file(self):
        if self._values['key_file'] in [None, 'none']:
            return None
        return self._values['key_file']

    @property
    def chain_file(self):
        if self._values['chain_file'] in [None, 'none']:
            return None
        return self._values['chain_file']

    @property
    def crl_file(self):
        if self._values['crl_file'] in [None, 'none']:
            return None
        return self._values['crl_file']

    @property
    def ciphers(self):
        if self._values['ciphers'] is None:
            return None
        if self._values['ciphers'] == 'none':
            return 'none'
        return self._values['ciphers'].split(' ')

    @property
    def modssl_methods(self):
        if self._values['modssl_methods'] is None:
            return None
        if self._values['modssl_methods'] == 'disabled':
            return 'no'
        return 'yes'

    @property
    def strict_resume(self):
        if self._values['strict_resume'] is None:
            return None
        if self._values['strict_resume'] == 'disabled':
            return 'no'
        return 'yes'

    @property
    def profile_mode_enabled(self):
        if self._values['profile_mode_enabled'] is None:
            return None
        if self._values['profile_mode_enabled'] == 'disabled':
            return 'no'
        return 'yes'

    @property
    def sni_require(self):
        if self._values['sni_require'] is None:
            return None
        if self._values['sni_require'] == 'false':
            return 'no'
        return 'yes'

    @property
    def sni_default(self):
        if self._values['sni_default'] is None:
            return None
        if self._values['sni_default'] == 'false':
            return 'no'
        return 'yes'