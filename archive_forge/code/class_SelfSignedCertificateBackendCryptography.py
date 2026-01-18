from __future__ import absolute_import, division, print_function
import os
from random import randrange
from ansible_collections.community.crypto.plugins.module_utils.crypto.support import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.cryptography_support import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.module_backends.certificate import (
class SelfSignedCertificateBackendCryptography(CertificateBackend):

    def __init__(self, module):
        super(SelfSignedCertificateBackendCryptography, self).__init__(module, 'cryptography')
        self.create_subject_key_identifier = module.params['selfsigned_create_subject_key_identifier']
        self.notBefore = get_relative_time_option(module.params['selfsigned_not_before'], 'selfsigned_not_before', backend=self.backend)
        self.notAfter = get_relative_time_option(module.params['selfsigned_not_after'], 'selfsigned_not_after', backend=self.backend)
        self.digest = select_message_digest(module.params['selfsigned_digest'])
        self.version = module.params['selfsigned_version']
        self.serial_number = x509.random_serial_number()
        if self.csr_path is not None and (not os.path.exists(self.csr_path)):
            raise CertificateError('The certificate signing request file {0} does not exist'.format(self.csr_path))
        if self.privatekey_content is None and (not os.path.exists(self.privatekey_path)):
            raise CertificateError('The private key file {0} does not exist'.format(self.privatekey_path))
        self._module = module
        self._ensure_private_key_loaded()
        self._ensure_csr_loaded()
        if self.csr is None:
            csr = cryptography.x509.CertificateSigningRequestBuilder()
            csr = csr.subject_name(cryptography.x509.Name([]))
            digest = None
            if cryptography_key_needs_digest_for_signing(self.privatekey):
                digest = self.digest
                if digest is None:
                    self.module.fail_json(msg='Unsupported digest "{0}"'.format(module.params['selfsigned_digest']))
            try:
                self.csr = csr.sign(self.privatekey, digest, default_backend())
            except TypeError as e:
                if str(e) == 'Algorithm must be a registered hash algorithm.' and digest is None:
                    self.module.fail_json(msg='Signing with Ed25519 and Ed448 keys requires cryptography 2.8 or newer.')
                raise
        if cryptography_key_needs_digest_for_signing(self.privatekey):
            if self.digest is None:
                raise CertificateError('The digest %s is not supported with the cryptography backend' % module.params['selfsigned_digest'])
        else:
            self.digest = None

    def generate_certificate(self):
        """(Re-)Generate certificate."""
        try:
            cert_builder = x509.CertificateBuilder()
            cert_builder = cert_builder.subject_name(self.csr.subject)
            cert_builder = cert_builder.issuer_name(self.csr.subject)
            cert_builder = cert_builder.serial_number(self.serial_number)
            cert_builder = cert_builder.not_valid_before(self.notBefore)
            cert_builder = cert_builder.not_valid_after(self.notAfter)
            cert_builder = cert_builder.public_key(self.privatekey.public_key())
            has_ski = False
            for extension in self.csr.extensions:
                if isinstance(extension.value, x509.SubjectKeyIdentifier):
                    if self.create_subject_key_identifier == 'always_create':
                        continue
                    has_ski = True
                cert_builder = cert_builder.add_extension(extension.value, critical=extension.critical)
            if not has_ski and self.create_subject_key_identifier != 'never_create':
                cert_builder = cert_builder.add_extension(x509.SubjectKeyIdentifier.from_public_key(self.privatekey.public_key()), critical=False)
        except ValueError as e:
            raise CertificateError(str(e))
        try:
            certificate = cert_builder.sign(private_key=self.privatekey, algorithm=self.digest, backend=default_backend())
        except TypeError as e:
            if str(e) == 'Algorithm must be a registered hash algorithm.' and self.digest is None:
                self.module.fail_json(msg='Signing with Ed25519 and Ed448 keys requires cryptography 2.8 or newer.')
            raise
        self.cert = certificate

    def get_certificate_data(self):
        """Return bytes for self.cert."""
        return self.cert.public_bytes(Encoding.PEM)

    def needs_regeneration(self):
        if super(SelfSignedCertificateBackendCryptography, self).needs_regeneration(not_before=self.notBefore, not_after=self.notAfter):
            return True
        self._ensure_existing_certificate_loaded()
        if not cryptography_verify_certificate_signature(self.existing_certificate, self.privatekey.public_key()):
            return True
        return False

    def dump(self, include_certificate):
        result = super(SelfSignedCertificateBackendCryptography, self).dump(include_certificate)
        if self.module.check_mode:
            result.update({'notBefore': self.notBefore.strftime('%Y%m%d%H%M%SZ'), 'notAfter': self.notAfter.strftime('%Y%m%d%H%M%SZ'), 'serial_number': self.serial_number})
        else:
            if self.cert is None:
                self.cert = self.existing_certificate
            result.update({'notBefore': self.cert.not_valid_before.strftime('%Y%m%d%H%M%SZ'), 'notAfter': self.cert.not_valid_after.strftime('%Y%m%d%H%M%SZ'), 'serial_number': cryptography_serial_number_of_cert(self.cert)})
        return result