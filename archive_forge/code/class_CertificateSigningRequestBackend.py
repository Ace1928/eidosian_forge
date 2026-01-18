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
@six.add_metaclass(abc.ABCMeta)
class CertificateSigningRequestBackend(object):

    def __init__(self, module, backend):
        self.module = module
        self.backend = backend
        self.digest = module.params['digest']
        self.privatekey_path = module.params['privatekey_path']
        self.privatekey_content = module.params['privatekey_content']
        if self.privatekey_content is not None:
            self.privatekey_content = self.privatekey_content.encode('utf-8')
        self.privatekey_passphrase = module.params['privatekey_passphrase']
        self.version = module.params['version']
        self.subjectAltName = module.params['subject_alt_name']
        self.subjectAltName_critical = module.params['subject_alt_name_critical']
        self.keyUsage = module.params['key_usage']
        self.keyUsage_critical = module.params['key_usage_critical']
        self.extendedKeyUsage = module.params['extended_key_usage']
        self.extendedKeyUsage_critical = module.params['extended_key_usage_critical']
        self.basicConstraints = module.params['basic_constraints']
        self.basicConstraints_critical = module.params['basic_constraints_critical']
        self.ocspMustStaple = module.params['ocsp_must_staple']
        self.ocspMustStaple_critical = module.params['ocsp_must_staple_critical']
        self.name_constraints_permitted = module.params['name_constraints_permitted'] or []
        self.name_constraints_excluded = module.params['name_constraints_excluded'] or []
        self.name_constraints_critical = module.params['name_constraints_critical']
        self.create_subject_key_identifier = module.params['create_subject_key_identifier']
        self.subject_key_identifier = module.params['subject_key_identifier']
        self.authority_key_identifier = module.params['authority_key_identifier']
        self.authority_cert_issuer = module.params['authority_cert_issuer']
        self.authority_cert_serial_number = module.params['authority_cert_serial_number']
        self.crl_distribution_points = module.params['crl_distribution_points']
        self.csr = None
        self.privatekey = None
        if self.create_subject_key_identifier and self.subject_key_identifier is not None:
            module.fail_json(msg='subject_key_identifier cannot be specified if create_subject_key_identifier is true')
        self.ordered_subject = False
        self.subject = [('C', module.params['country_name']), ('ST', module.params['state_or_province_name']), ('L', module.params['locality_name']), ('O', module.params['organization_name']), ('OU', module.params['organizational_unit_name']), ('CN', module.params['common_name']), ('emailAddress', module.params['email_address'])]
        self.subject = [(entry[0], entry[1]) for entry in self.subject if entry[1]]
        try:
            if module.params['subject']:
                self.subject = self.subject + parse_name_field(module.params['subject'], 'subject')
            if module.params['subject_ordered']:
                if self.subject:
                    raise CertificateSigningRequestError('subject_ordered cannot be combined with any other subject field')
                self.subject = parse_ordered_name_field(module.params['subject_ordered'], 'subject_ordered')
                self.ordered_subject = True
        except ValueError as exc:
            raise CertificateSigningRequestError(to_native(exc))
        self.using_common_name_for_san = False
        if not self.subjectAltName and module.params['use_common_name_for_san']:
            for sub in self.subject:
                if sub[0] in ('commonName', 'CN'):
                    self.subjectAltName = ['DNS:%s' % sub[1]]
                    self.using_common_name_for_san = True
                    break
        if self.subject_key_identifier is not None:
            try:
                self.subject_key_identifier = binascii.unhexlify(self.subject_key_identifier.replace(':', ''))
            except Exception as e:
                raise CertificateSigningRequestError('Cannot parse subject_key_identifier: {0}'.format(e))
        if self.authority_key_identifier is not None:
            try:
                self.authority_key_identifier = binascii.unhexlify(self.authority_key_identifier.replace(':', ''))
            except Exception as e:
                raise CertificateSigningRequestError('Cannot parse authority_key_identifier: {0}'.format(e))
        self.existing_csr = None
        self.existing_csr_bytes = None
        self.diff_before = self._get_info(None)
        self.diff_after = self._get_info(None)

    def _get_info(self, data):
        if data is None:
            return dict()
        try:
            result = get_csr_info(self.module, self.backend, data, validate_signature=False, prefer_one_fingerprint=True)
            result['can_parse_csr'] = True
            return result
        except Exception as exc:
            return dict(can_parse_csr=False)

    @abc.abstractmethod
    def generate_csr(self):
        """(Re-)Generate CSR."""
        pass

    @abc.abstractmethod
    def get_csr_data(self):
        """Return bytes for self.csr."""
        pass

    def set_existing(self, csr_bytes):
        """Set existing CSR bytes. None indicates that the CSR does not exist."""
        self.existing_csr_bytes = csr_bytes
        self.diff_after = self.diff_before = self._get_info(self.existing_csr_bytes)

    def has_existing(self):
        """Query whether an existing CSR is/has been there."""
        return self.existing_csr_bytes is not None

    def _ensure_private_key_loaded(self):
        """Load the provided private key into self.privatekey."""
        if self.privatekey is not None:
            return
        try:
            self.privatekey = load_privatekey(path=self.privatekey_path, content=self.privatekey_content, passphrase=self.privatekey_passphrase, backend=self.backend)
        except OpenSSLBadPassphraseError as exc:
            raise CertificateSigningRequestError(exc)

    @abc.abstractmethod
    def _check_csr(self):
        """Check whether provided parameters, assuming self.existing_csr and self.privatekey have been populated."""
        pass

    def needs_regeneration(self):
        """Check whether a regeneration is necessary."""
        if self.existing_csr_bytes is None:
            return True
        try:
            self.existing_csr = load_certificate_request(None, content=self.existing_csr_bytes, backend=self.backend)
        except Exception as dummy:
            return True
        self._ensure_private_key_loaded()
        return not self._check_csr()

    def dump(self, include_csr):
        """Serialize the object into a dictionary."""
        result = {'privatekey': self.privatekey_path, 'subject': self.subject, 'subjectAltName': self.subjectAltName, 'keyUsage': self.keyUsage, 'extendedKeyUsage': self.extendedKeyUsage, 'basicConstraints': self.basicConstraints, 'ocspMustStaple': self.ocspMustStaple, 'name_constraints_permitted': self.name_constraints_permitted, 'name_constraints_excluded': self.name_constraints_excluded}
        csr_bytes = self.existing_csr_bytes
        if self.csr is not None:
            csr_bytes = self.get_csr_data()
        self.diff_after = self._get_info(csr_bytes)
        if include_csr:
            result['csr'] = csr_bytes.decode('utf-8') if csr_bytes else None
        result['diff'] = dict(before=self.diff_before, after=self.diff_after)
        return result