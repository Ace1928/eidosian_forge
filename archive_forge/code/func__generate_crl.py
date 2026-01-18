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
def _generate_crl(self):
    backend = default_backend()
    crl = CertificateRevocationListBuilder()
    try:
        crl = crl.issuer_name(Name([NameAttribute(cryptography_name_to_oid(entry[0]), to_text(entry[1])) for entry in self.issuer]))
    except ValueError as e:
        raise CRLError(e)
    crl = crl.last_update(self.last_update)
    crl = crl.next_update(self.next_update)
    if self.update and self.crl:
        new_entries = set([self._compress_entry(entry) for entry in self.revoked_certificates])
        for entry in self.crl:
            decoded_entry = self._compress_entry(cryptography_decode_revoked_certificate(entry))
            if decoded_entry not in new_entries:
                crl = crl.add_revoked_certificate(entry)
    for entry in self.revoked_certificates:
        revoked_cert = RevokedCertificateBuilder()
        revoked_cert = revoked_cert.serial_number(entry['serial_number'])
        revoked_cert = revoked_cert.revocation_date(entry['revocation_date'])
        if entry['issuer'] is not None:
            revoked_cert = revoked_cert.add_extension(x509.CertificateIssuer(entry['issuer']), entry['issuer_critical'])
        if entry['reason'] is not None:
            revoked_cert = revoked_cert.add_extension(x509.CRLReason(entry['reason']), entry['reason_critical'])
        if entry['invalidity_date'] is not None:
            revoked_cert = revoked_cert.add_extension(x509.InvalidityDate(entry['invalidity_date']), entry['invalidity_date_critical'])
        crl = crl.add_revoked_certificate(revoked_cert.build(backend))
    digest = None
    if cryptography_key_needs_digest_for_signing(self.privatekey):
        digest = self.digest
    self.crl = crl.sign(self.privatekey, digest, backend=backend)
    if self.format == 'pem':
        return self.crl.public_bytes(Encoding.PEM)
    else:
        return self.crl.public_bytes(Encoding.DER)