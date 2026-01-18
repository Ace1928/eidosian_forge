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
def _get_ec_class(self, ectype):
    ecclass = cryptography.hazmat.primitives.asymmetric.ec.__dict__.get(ectype)
    if ecclass is None:
        self.module.fail_json(msg='Your cryptography version does not support {0}'.format(ectype))
    return ecclass