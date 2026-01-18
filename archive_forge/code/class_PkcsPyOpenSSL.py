from __future__ import absolute_import, division, print_function
import abc
import base64
import os
import stat
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common.text.converters import to_bytes, to_native
from ansible_collections.community.crypto.plugins.module_utils.version import LooseVersion
from ansible_collections.community.crypto.plugins.module_utils.io import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.basic import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.cryptography_support import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.support import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.pem import (
class PkcsPyOpenSSL(Pkcs):

    def __init__(self, module):
        super(PkcsPyOpenSSL, self).__init__(module, 'pyopenssl')
        if self.encryption_level != 'auto':
            module.fail_json(msg='The PyOpenSSL backend only supports encryption_level = auto')

    def generate_bytes(self, module):
        """Generate PKCS#12 file archive."""
        self.pkcs12 = crypto.PKCS12()
        if self.other_certificates:
            self.pkcs12.set_ca_certificates(self.other_certificates)
        if self.certificate_path:
            self.pkcs12.set_certificate(load_certificate(self.certificate_path, backend=self.backend))
        if self.friendly_name:
            self.pkcs12.set_friendlyname(to_bytes(self.friendly_name))
        if self.privatekey_content:
            try:
                self.pkcs12.set_privatekey(load_privatekey(None, content=self.privatekey_content, passphrase=self.privatekey_passphrase, backend=self.backend))
            except OpenSSLBadPassphraseError as exc:
                raise PkcsError(exc)
        return self.pkcs12.export(self.passphrase, self.iter_size, self.maciter_size)

    def parse_bytes(self, pkcs12_content):
        try:
            p12 = crypto.load_pkcs12(pkcs12_content, self.passphrase)
            pkey = p12.get_privatekey()
            if pkey is not None:
                pkey = crypto.dump_privatekey(crypto.FILETYPE_PEM, pkey)
            crt = p12.get_certificate()
            if crt is not None:
                crt = crypto.dump_certificate(crypto.FILETYPE_PEM, crt)
            other_certs = []
            if p12.get_ca_certificates() is not None:
                other_certs = [crypto.dump_certificate(crypto.FILETYPE_PEM, other_cert) for other_cert in p12.get_ca_certificates()]
            friendly_name = p12.get_friendlyname()
            return (pkey, crt, other_certs, friendly_name)
        except crypto.Error as exc:
            raise PkcsError(exc)

    def _dump_privatekey(self, pkcs12):
        pk = pkcs12.get_privatekey()
        return crypto.dump_privatekey(crypto.FILETYPE_PEM, pk) if pk else None

    def _dump_certificate(self, pkcs12):
        cert = pkcs12.get_certificate()
        return crypto.dump_certificate(crypto.FILETYPE_PEM, cert) if cert else None

    def _dump_other_certificates(self, pkcs12):
        if pkcs12.get_ca_certificates() is None:
            return []
        return [crypto.dump_certificate(crypto.FILETYPE_PEM, other_cert) for other_cert in pkcs12.get_ca_certificates()]

    def _get_friendly_name(self, pkcs12):
        return pkcs12.get_friendlyname()