from __future__ import absolute_import, division, print_function
import abc
import binascii
import os
from base64 import b64encode
from datetime import datetime
from hashlib import sha256
from ansible.module_utils import six
from ansible.module_utils.common.text.converters import to_text
from ansible_collections.community.crypto.plugins.module_utils.crypto.support import convert_relative_to_datetime
from ansible_collections.community.crypto.plugins.module_utils.openssh.utils import (
@staticmethod
def _parse_cert_info(pub_key_type, parser):
    cert_info = get_cert_info_object(pub_key_type)
    cert_info.nonce = parser.string()
    cert_info.parse_public_numbers(parser)
    cert_info.serial = parser.uint64()
    cert_info.cert_type = parser.uint32()
    cert_info.key_id = parser.string()
    cert_info.principals = parser.string_list()
    cert_info.valid_after = parser.uint64()
    cert_info.valid_before = parser.uint64()
    cert_info.critical_options = parser.option_list()
    cert_info.extensions = parser.option_list()
    cert_info.reserved = parser.string()
    cert_info.signing_key = parser.string()
    return cert_info