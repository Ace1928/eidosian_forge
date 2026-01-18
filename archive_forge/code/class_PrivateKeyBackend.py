from __future__ import absolute_import, division, print_function
import abc
import base64
import traceback
from ansible.module_utils import six
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.common.text.converters import to_bytes
from ansible_collections.community.crypto.plugins.module_utils.version import LooseVersion
from ansible_collections.community.crypto.plugins.module_utils.crypto.basic import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.support import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.pem import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.module_backends.privatekey_info import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.module_backends.common import ArgumentSpec
@six.add_metaclass(abc.ABCMeta)
class PrivateKeyBackend:

    def __init__(self, module, backend):
        self.module = module
        self.type = module.params['type']
        self.size = module.params['size']
        self.curve = module.params['curve']
        self.passphrase = module.params['passphrase']
        self.cipher = module.params['cipher']
        self.format = module.params['format']
        self.format_mismatch = module.params.get('format_mismatch', 'regenerate')
        self.regenerate = module.params.get('regenerate', 'full_idempotence')
        self.backend = backend
        self.private_key = None
        self.existing_private_key = None
        self.existing_private_key_bytes = None
        self.diff_before = self._get_info(None)
        self.diff_after = self._get_info(None)

    def _get_info(self, data):
        if data is None:
            return dict()
        result = dict(can_parse_key=False)
        try:
            result.update(get_privatekey_info(self.module, self.backend, data, passphrase=self.passphrase, return_private_key_data=False, prefer_one_fingerprint=True))
        except PrivateKeyConsistencyError as exc:
            result.update(exc.result)
        except PrivateKeyParseError as exc:
            result.update(exc.result)
        except Exception as exc:
            pass
        return result

    @abc.abstractmethod
    def generate_private_key(self):
        """(Re-)Generate private key."""
        pass

    def convert_private_key(self):
        """Convert existing private key (self.existing_private_key) to new private key (self.private_key).

        This is effectively a copy without active conversion. The conversion is done
        during load and store; get_private_key_data() uses the destination format to
        serialize the key.
        """
        self._ensure_existing_private_key_loaded()
        self.private_key = self.existing_private_key

    @abc.abstractmethod
    def get_private_key_data(self):
        """Return bytes for self.private_key."""
        pass

    def set_existing(self, privatekey_bytes):
        """Set existing private key bytes. None indicates that the key does not exist."""
        self.existing_private_key_bytes = privatekey_bytes
        self.diff_after = self.diff_before = self._get_info(self.existing_private_key_bytes)

    def has_existing(self):
        """Query whether an existing private key is/has been there."""
        return self.existing_private_key_bytes is not None

    @abc.abstractmethod
    def _check_passphrase(self):
        """Check whether provided passphrase matches, assuming self.existing_private_key_bytes has been populated."""
        pass

    @abc.abstractmethod
    def _ensure_existing_private_key_loaded(self):
        """Make sure that self.existing_private_key is populated from self.existing_private_key_bytes."""
        pass

    @abc.abstractmethod
    def _check_size_and_type(self):
        """Check whether provided size and type matches, assuming self.existing_private_key has been populated."""
        pass

    @abc.abstractmethod
    def _check_format(self):
        """Check whether the key file format, assuming self.existing_private_key and self.existing_private_key_bytes has been populated."""
        pass

    def needs_regeneration(self):
        """Check whether a regeneration is necessary."""
        if self.regenerate == 'always':
            return True
        if not self.has_existing():
            return True
        if not self._check_passphrase():
            if self.regenerate == 'full_idempotence':
                return True
            self.module.fail_json(msg='Unable to read the key. The key is protected with a another passphrase / no passphrase or broken. Will not proceed. To force regeneration, call the module with `generate` set to `full_idempotence` or `always`, or with `force=true`.')
        self._ensure_existing_private_key_loaded()
        if self.regenerate != 'never':
            if not self._check_size_and_type():
                if self.regenerate in ('partial_idempotence', 'full_idempotence'):
                    return True
                self.module.fail_json(msg='Key has wrong type and/or size. Will not proceed. To force regeneration, call the module with `generate` set to `partial_idempotence`, `full_idempotence` or `always`, or with `force=true`.')
        if self.format_mismatch == 'regenerate' and self.regenerate != 'never':
            if not self._check_format():
                if self.regenerate in ('partial_idempotence', 'full_idempotence'):
                    return True
                self.module.fail_json(msg='Key has wrong format. Will not proceed. To force regeneration, call the module with `generate` set to `partial_idempotence`, `full_idempotence` or `always`, or with `force=true`. To convert the key, set `format_mismatch` to `convert`.')
        return False

    def needs_conversion(self):
        """Check whether a conversion is necessary. Must only be called if needs_regeneration() returned False."""
        self._ensure_existing_private_key_loaded()
        return self.has_existing() and self.format_mismatch == 'convert' and (not self._check_format())

    def _get_fingerprint(self):
        if self.private_key:
            return get_fingerprint_of_privatekey(self.private_key, backend=self.backend)
        try:
            self._ensure_existing_private_key_loaded()
        except Exception as dummy:
            pass
        if self.existing_private_key:
            return get_fingerprint_of_privatekey(self.existing_private_key, backend=self.backend)

    def dump(self, include_key):
        """Serialize the object into a dictionary."""
        if not self.private_key:
            try:
                self._ensure_existing_private_key_loaded()
            except Exception as dummy:
                pass
        result = {'type': self.type, 'size': self.size, 'fingerprint': self._get_fingerprint()}
        if self.type == 'ECC':
            result['curve'] = self.curve
        pk_bytes = self.existing_private_key_bytes
        if self.private_key is not None:
            pk_bytes = self.get_private_key_data()
        self.diff_after = self._get_info(pk_bytes)
        if include_key:
            if pk_bytes:
                if identify_private_key_format(pk_bytes) == 'raw':
                    result['privatekey'] = base64.b64encode(pk_bytes)
                else:
                    result['privatekey'] = pk_bytes.decode('utf-8')
            else:
                result['privatekey'] = None
        result['diff'] = dict(before=self.diff_before, after=self.diff_after)
        return result