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
class OpensshECDSACertificateInfo(OpensshCertificateInfo):

    def __init__(self, curve=None, public_key=None, **kwargs):
        super(OpensshECDSACertificateInfo, self).__init__(**kwargs)
        self._curve = None
        if curve is not None:
            self.curve = curve
        self.public_key = public_key

    @property
    def curve(self):
        return self._curve

    @curve.setter
    def curve(self, curve):
        if curve in _ECDSA_CURVE_IDENTIFIERS.values():
            self._curve = curve
            self.type_string = _SSH_TYPE_STRINGS[_ECDSA_CURVE_IDENTIFIERS_LOOKUP[curve]] + _CERT_SUFFIX_V01
        else:
            raise ValueError('Curve must be one of %s' % b','.join(list(_ECDSA_CURVE_IDENTIFIERS.values())).decode('UTF-8'))

    def public_key_fingerprint(self):
        if any([self.curve is None, self.public_key is None]):
            return b''
        writer = _OpensshWriter()
        writer.string(_SSH_TYPE_STRINGS[_ECDSA_CURVE_IDENTIFIERS_LOOKUP[self.curve]])
        writer.string(self.curve)
        writer.string(self.public_key)
        return fingerprint(writer.bytes())

    def parse_public_numbers(self, parser):
        self.curve = parser.string()
        self.public_key = parser.string()