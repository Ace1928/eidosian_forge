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
class OpensshCertificate(object):
    """Encapsulates a formatted OpenSSH certificate including signature and signing key"""

    def __init__(self, cert_info, signature):
        self._cert_info = cert_info
        self.signature = signature

    @classmethod
    def load(cls, path):
        if not os.path.exists(path):
            raise ValueError('%s is not a valid path.' % path)
        try:
            with open(path, 'rb') as cert_file:
                data = cert_file.read()
        except (IOError, OSError) as e:
            raise ValueError('%s cannot be opened for reading: %s' % (path, e))
        try:
            format_identifier, b64_cert = data.split(b' ')[:2]
            cert = binascii.a2b_base64(b64_cert)
        except (binascii.Error, ValueError):
            raise ValueError('Certificate not in OpenSSH format')
        for key_type, string in _SSH_TYPE_STRINGS.items():
            if format_identifier == string + _CERT_SUFFIX_V01:
                pub_key_type = key_type
                break
        else:
            raise ValueError('Invalid certificate format identifier: %s' % format_identifier)
        parser = OpensshParser(cert)
        if format_identifier != parser.string():
            raise ValueError('Certificate formats do not match')
        try:
            cert_info = cls._parse_cert_info(pub_key_type, parser)
            signature = parser.string()
        except (TypeError, ValueError) as e:
            raise ValueError('Invalid certificate data: %s' % e)
        if parser.remaining_bytes():
            raise ValueError('%s bytes of additional data was not parsed while loading %s' % (parser.remaining_bytes(), path))
        return cls(cert_info=cert_info, signature=signature)

    @property
    def type_string(self):
        return to_text(self._cert_info.type_string)

    @property
    def nonce(self):
        return self._cert_info.nonce

    @property
    def public_key(self):
        return to_text(self._cert_info.public_key_fingerprint())

    @property
    def serial(self):
        return self._cert_info.serial

    @property
    def type(self):
        return self._cert_info.cert_type

    @property
    def key_id(self):
        return to_text(self._cert_info.key_id)

    @property
    def principals(self):
        return [to_text(p) for p in self._cert_info.principals]

    @property
    def valid_after(self):
        return self._cert_info.valid_after

    @property
    def valid_before(self):
        return self._cert_info.valid_before

    @property
    def critical_options(self):
        return [OpensshCertificateOption('critical', to_text(n), to_text(d)) for n, d in self._cert_info.critical_options]

    @property
    def extensions(self):
        return [OpensshCertificateOption('extension', to_text(n), to_text(d)) for n, d in self._cert_info.extensions]

    @property
    def reserved(self):
        return self._cert_info.reserved

    @property
    def signing_key(self):
        return to_text(self._cert_info.signing_key_fingerprint())

    @property
    def signature_type(self):
        signature_data = OpensshParser.signature_data(self.signature)
        return to_text(signature_data['signature_type'])

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

    def to_dict(self):
        time_parameters = OpensshCertificateTimeParameters(valid_from=self.valid_after, valid_to=self.valid_before)
        return {'type_string': self.type_string, 'nonce': self.nonce, 'serial': self.serial, 'cert_type': self.type, 'identifier': self.key_id, 'principals': self.principals, 'valid_after': time_parameters.valid_from(date_format='human_readable'), 'valid_before': time_parameters.valid_to(date_format='human_readable'), 'critical_options': [str(critical_option) for critical_option in self.critical_options], 'extensions': [str(extension) for extension in self.extensions], 'reserved': self.reserved, 'public_key': self.public_key, 'signing_key': self.signing_key}