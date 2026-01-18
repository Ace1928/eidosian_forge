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
def _parse_serial_number(self, value, index):
    if self.serial_numbers_format == 'integer':
        try:
            return check_type_int(value)
        except TypeError as exc:
            self.module.fail_json(msg='Error while parsing revoked_certificates[{idx}].serial_number as an integer: {exc}'.format(idx=index + 1, exc=to_native(exc)))
    if self.serial_numbers_format == 'hex-octets':
        try:
            return parse_serial(check_type_str(value))
        except (TypeError, ValueError) as exc:
            self.module.fail_json(msg='Error while parsing revoked_certificates[{idx}].serial_number as an colon-separated hex octet string: {exc}'.format(idx=index + 1, exc=to_native(exc)))
    raise RuntimeError('Unexpected value %s of serial_numbers' % (self.serial_numbers_format,))