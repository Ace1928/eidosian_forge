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
class OpensshDSACertificateInfo(OpensshCertificateInfo):

    def __init__(self, p=None, q=None, g=None, y=None, **kwargs):
        super(OpensshDSACertificateInfo, self).__init__(**kwargs)
        self.type_string = _SSH_TYPE_STRINGS['dsa'] + _CERT_SUFFIX_V01
        self.p = p
        self.q = q
        self.g = g
        self.y = y

    def public_key_fingerprint(self):
        if any([self.p is None, self.q is None, self.g is None, self.y is None]):
            return b''
        writer = _OpensshWriter()
        writer.string(_SSH_TYPE_STRINGS['dsa'])
        writer.mpint(self.p)
        writer.mpint(self.q)
        writer.mpint(self.g)
        writer.mpint(self.y)
        return fingerprint(writer.bytes())

    def parse_public_numbers(self, parser):
        self.p = parser.mpint()
        self.q = parser.mpint()
        self.g = parser.mpint()
        self.y = parser.mpint()