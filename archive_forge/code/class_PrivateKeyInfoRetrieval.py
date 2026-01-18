from __future__ import absolute_import, division, print_function
import abc
import traceback
from ansible.module_utils import six
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.common.text.converters import to_native, to_bytes
from ansible_collections.community.crypto.plugins.module_utils.version import LooseVersion
from ansible_collections.community.crypto.plugins.module_utils.crypto.basic import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.support import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.math import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.module_backends.publickey_info import (
@six.add_metaclass(abc.ABCMeta)
class PrivateKeyInfoRetrieval(object):

    def __init__(self, module, backend, content, passphrase=None, return_private_key_data=False, check_consistency=False):
        self.module = module
        self.backend = backend
        self.content = content
        self.passphrase = passphrase
        self.return_private_key_data = return_private_key_data
        self.check_consistency = check_consistency

    @abc.abstractmethod
    def _get_public_key(self, binary):
        pass

    @abc.abstractmethod
    def _get_key_info(self, need_private_key_data=False):
        pass

    @abc.abstractmethod
    def _is_key_consistent(self, key_public_data, key_private_data):
        pass

    def get_info(self, prefer_one_fingerprint=False):
        result = dict(can_parse_key=False, key_is_consistent=None)
        priv_key_detail = self.content
        try:
            self.key = load_privatekey(path=None, content=priv_key_detail, passphrase=to_bytes(self.passphrase) if self.passphrase is not None else self.passphrase, backend=self.backend)
            result['can_parse_key'] = True
        except OpenSSLObjectError as exc:
            raise PrivateKeyParseError(to_native(exc), result)
        result['public_key'] = to_native(self._get_public_key(binary=False))
        pk = self._get_public_key(binary=True)
        result['public_key_fingerprints'] = get_fingerprint_of_bytes(pk, prefer_one=prefer_one_fingerprint) if pk is not None else dict()
        key_type, key_public_data, key_private_data = self._get_key_info(need_private_key_data=self.return_private_key_data or self.check_consistency)
        result['type'] = key_type
        result['public_data'] = key_public_data
        if self.return_private_key_data:
            result['private_data'] = key_private_data
        if self.check_consistency:
            result['key_is_consistent'] = self._is_key_consistent(key_public_data, key_private_data)
            if result['key_is_consistent'] is False:
                msg = 'Private key is not consistent! (See https://blog.hboeck.de/archives/888-How-I-tricked-Symantec-with-a-Fake-Private-Key.html)'
                raise PrivateKeyConsistencyError(msg, result)
        return result