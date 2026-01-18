from __future__ import absolute_import, division, print_function
import abc
import traceback
from ansible.module_utils import six
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.crypto.plugins.module_utils.version import LooseVersion
from ansible_collections.community.crypto.plugins.module_utils.crypto.basic import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.support import (
@six.add_metaclass(abc.ABCMeta)
class PublicKeyInfoRetrieval(object):

    def __init__(self, module, backend, content=None, key=None):
        self.module = module
        self.backend = backend
        self.content = content
        self.key = key

    @abc.abstractmethod
    def _get_public_key(self, binary):
        pass

    @abc.abstractmethod
    def _get_key_info(self):
        pass

    def get_info(self, prefer_one_fingerprint=False):
        result = dict()
        if self.key is None:
            try:
                self.key = load_publickey(content=self.content, backend=self.backend)
            except OpenSSLObjectError as e:
                raise PublicKeyParseError(to_native(e), {})
        pk = self._get_public_key(binary=True)
        result['fingerprints'] = get_fingerprint_of_bytes(pk, prefer_one=prefer_one_fingerprint) if pk is not None else dict()
        key_type, key_public_data = self._get_key_info()
        result['type'] = key_type
        result['public_data'] = key_public_data
        return result