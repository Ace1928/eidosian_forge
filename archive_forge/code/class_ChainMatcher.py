from __future__ import absolute_import, division, print_function
import abc
from ansible.module_utils import six
from ansible_collections.community.crypto.plugins.module_utils.acme.errors import (
from ansible_collections.community.crypto.plugins.module_utils.acme.utils import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.pem import (
@six.add_metaclass(abc.ABCMeta)
class ChainMatcher(object):

    @abc.abstractmethod
    def match(self, certificate):
        """
        Check whether a certificate chain (CertificateChain instance) matches.
        """