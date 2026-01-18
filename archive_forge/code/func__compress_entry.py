from __future__ import absolute_import, division, print_function
import base64
import os
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible.module_utils.common.validation import check_type_int, check_type_str
from ansible_collections.community.crypto.plugins.module_utils.serial import parse_serial
from ansible_collections.community.crypto.plugins.module_utils.version import LooseVersion
from ansible_collections.community.crypto.plugins.module_utils.io import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.basic import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.support import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.cryptography_support import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.cryptography_crl import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.pem import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.module_backends.crl_info import (
def _compress_entry(self, entry):
    issuer = None
    if entry['issuer'] is not None:
        issuer = tuple((cryptography_decode_name(issuer, idn_rewrite='idna') for issuer in entry['issuer']))
    if self.ignore_timestamps:
        return (entry['serial_number'], issuer, entry['issuer_critical'], entry['reason'], entry['reason_critical'], entry['invalidity_date'], entry['invalidity_date_critical'])
    else:
        return (entry['serial_number'], entry['revocation_date'], issuer, entry['issuer_critical'], entry['reason'], entry['reason_critical'], entry['invalidity_date'], entry['invalidity_date_critical'])