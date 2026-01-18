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
class OpensshED25519CertificateInfo(OpensshCertificateInfo):

    def __init__(self, pk=None, **kwargs):
        super(OpensshED25519CertificateInfo, self).__init__(**kwargs)
        self.type_string = _SSH_TYPE_STRINGS['ed25519'] + _CERT_SUFFIX_V01
        self.pk = pk

    def public_key_fingerprint(self):
        if self.pk is None:
            return b''
        writer = _OpensshWriter()
        writer.string(_SSH_TYPE_STRINGS['ed25519'])
        writer.string(self.pk)
        return fingerprint(writer.bytes())

    def parse_public_numbers(self, parser):
        self.pk = parser.string()