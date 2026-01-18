from __future__ import absolute_import, division, print_function
import abc
import binascii
import traceback
from ansible.module_utils import six
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible_collections.community.crypto.plugins.module_utils.version import LooseVersion
from ansible_collections.community.crypto.plugins.module_utils.crypto.basic import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.support import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.cryptography_support import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.cryptography_crl import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.module_backends.csr_info import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.module_backends.common import ArgumentSpec
def _check_csr(self):
    """Check whether provided parameters, assuming self.existing_csr and self.privatekey have been populated."""

    def _check_subject(csr):
        subject = [(cryptography_name_to_oid(entry[0]), to_text(entry[1])) for entry in self.subject]
        current_subject = [(sub.oid, sub.value) for sub in csr.subject]
        if self.ordered_subject:
            return subject == current_subject
        else:
            return set(subject) == set(current_subject)

    def _find_extension(extensions, exttype):
        return next((ext for ext in extensions if isinstance(ext.value, exttype)), None)

    def _check_subjectAltName(extensions):
        current_altnames_ext = _find_extension(extensions, cryptography.x509.SubjectAlternativeName)
        current_altnames = [to_text(altname) for altname in current_altnames_ext.value] if current_altnames_ext else []
        altnames = [to_text(cryptography_get_name(altname)) for altname in self.subjectAltName] if self.subjectAltName else []
        if set(altnames) != set(current_altnames):
            return False
        if altnames:
            if current_altnames_ext.critical != self.subjectAltName_critical:
                return False
        return True

    def _check_keyUsage(extensions):
        current_keyusage_ext = _find_extension(extensions, cryptography.x509.KeyUsage)
        if not self.keyUsage:
            return current_keyusage_ext is None
        elif current_keyusage_ext is None:
            return False
        params = cryptography_parse_key_usage_params(self.keyUsage)
        for param in params:
            if getattr(current_keyusage_ext.value, '_' + param) != params[param]:
                return False
        if current_keyusage_ext.critical != self.keyUsage_critical:
            return False
        return True

    def _check_extenededKeyUsage(extensions):
        current_usages_ext = _find_extension(extensions, cryptography.x509.ExtendedKeyUsage)
        current_usages = [str(usage) for usage in current_usages_ext.value] if current_usages_ext else []
        usages = [str(cryptography_name_to_oid(usage)) for usage in self.extendedKeyUsage] if self.extendedKeyUsage else []
        if set(current_usages) != set(usages):
            return False
        if usages:
            if current_usages_ext.critical != self.extendedKeyUsage_critical:
                return False
        return True

    def _check_basicConstraints(extensions):
        bc_ext = _find_extension(extensions, cryptography.x509.BasicConstraints)
        current_ca = bc_ext.value.ca if bc_ext else False
        current_path_length = bc_ext.value.path_length if bc_ext else None
        ca, path_length = cryptography_get_basic_constraints(self.basicConstraints)
        if ca != current_ca:
            return False
        if path_length != current_path_length:
            return False
        if self.basicConstraints:
            return bc_ext is not None and bc_ext.critical == self.basicConstraints_critical
        else:
            return bc_ext is None

    def _check_ocspMustStaple(extensions):
        try:
            tlsfeature_ext = _find_extension(extensions, cryptography.x509.TLSFeature)
            has_tlsfeature = True
        except AttributeError as dummy:
            tlsfeature_ext = next((ext for ext in extensions if ext.value.oid == CRYPTOGRAPHY_MUST_STAPLE_NAME), None)
            has_tlsfeature = False
        if self.ocspMustStaple:
            if not tlsfeature_ext or tlsfeature_ext.critical != self.ocspMustStaple_critical:
                return False
            if has_tlsfeature:
                return cryptography.x509.TLSFeatureType.status_request in tlsfeature_ext.value
            else:
                return tlsfeature_ext.value.value == CRYPTOGRAPHY_MUST_STAPLE_VALUE
        else:
            return tlsfeature_ext is None

    def _check_nameConstraints(extensions):
        current_nc_ext = _find_extension(extensions, cryptography.x509.NameConstraints)
        current_nc_perm = [to_text(altname) for altname in current_nc_ext.value.permitted_subtrees or []] if current_nc_ext else []
        current_nc_excl = [to_text(altname) for altname in current_nc_ext.value.excluded_subtrees or []] if current_nc_ext else []
        nc_perm = [to_text(cryptography_get_name(altname, 'name constraints permitted')) for altname in self.name_constraints_permitted]
        nc_excl = [to_text(cryptography_get_name(altname, 'name constraints excluded')) for altname in self.name_constraints_excluded]
        if set(nc_perm) != set(current_nc_perm) or set(nc_excl) != set(current_nc_excl):
            return False
        if nc_perm or nc_excl:
            if current_nc_ext.critical != self.name_constraints_critical:
                return False
        return True

    def _check_subject_key_identifier(extensions):
        ext = _find_extension(extensions, cryptography.x509.SubjectKeyIdentifier)
        if self.create_subject_key_identifier or self.subject_key_identifier is not None:
            if not ext or ext.critical:
                return False
            if self.create_subject_key_identifier:
                digest = cryptography.x509.SubjectKeyIdentifier.from_public_key(self.privatekey.public_key()).digest
                return ext.value.digest == digest
            else:
                return ext.value.digest == self.subject_key_identifier
        else:
            return ext is None

    def _check_authority_key_identifier(extensions):
        ext = _find_extension(extensions, cryptography.x509.AuthorityKeyIdentifier)
        if self.authority_key_identifier is not None or self.authority_cert_issuer is not None or self.authority_cert_serial_number is not None:
            if not ext or ext.critical:
                return False
            aci = None
            csr_aci = None
            if self.authority_cert_issuer is not None:
                aci = [to_text(cryptography_get_name(n, 'authority cert issuer')) for n in self.authority_cert_issuer]
            if ext.value.authority_cert_issuer is not None:
                csr_aci = [to_text(n) for n in ext.value.authority_cert_issuer]
            return ext.value.key_identifier == self.authority_key_identifier and csr_aci == aci and (ext.value.authority_cert_serial_number == self.authority_cert_serial_number)
        else:
            return ext is None

    def _check_crl_distribution_points(extensions):
        ext = _find_extension(extensions, cryptography.x509.CRLDistributionPoints)
        if self.crl_distribution_points is None:
            return ext is None
        if not ext:
            return False
        return list(ext.value) == self.crl_distribution_points

    def _check_extensions(csr):
        extensions = csr.extensions
        return _check_subjectAltName(extensions) and _check_keyUsage(extensions) and _check_extenededKeyUsage(extensions) and _check_basicConstraints(extensions) and _check_ocspMustStaple(extensions) and _check_subject_key_identifier(extensions) and _check_authority_key_identifier(extensions) and _check_nameConstraints(extensions) and _check_crl_distribution_points(extensions)

    def _check_signature(csr):
        if not csr.is_signature_valid:
            return False
        key_a = csr.public_key().public_bytes(cryptography.hazmat.primitives.serialization.Encoding.PEM, cryptography.hazmat.primitives.serialization.PublicFormat.SubjectPublicKeyInfo)
        key_b = self.privatekey.public_key().public_bytes(cryptography.hazmat.primitives.serialization.Encoding.PEM, cryptography.hazmat.primitives.serialization.PublicFormat.SubjectPublicKeyInfo)
        return key_a == key_b
    return _check_subject(self.existing_csr) and _check_extensions(self.existing_csr) and _check_signature(self.existing_csr)