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
def _add_curve(self, name, ectype, deprecated=False):

    def create(size):
        ecclass = self._get_ec_class(ectype)
        return ecclass()

    def verify(privatekey):
        ecclass = self._get_ec_class(ectype)
        return isinstance(privatekey.private_numbers().public_numbers.curve, ecclass)
    self.curves[name] = {'create': create, 'verify': verify, 'deprecated': deprecated}