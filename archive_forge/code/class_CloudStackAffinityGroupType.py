import base64
import warnings
from libcloud.utils.py3 import b, urlparse
from libcloud.compute.base import (
from libcloud.compute.types import (
from libcloud.utils.networking import is_private_subnet
from libcloud.common.cloudstack import CloudStackDriverMixIn
from libcloud.compute.providers import Provider
class CloudStackAffinityGroupType:
    """
    Class representing a CloudStack AffinityGroupType.
    """

    def __init__(self, type_name):
        """
        A CloudStack Affinity Group Type.

        @note: This is a non-standard extension API, and only works for
               CloudStack.

        :param      type_name: the type of the affinity group
        :type       type_name: ``str``

        :rtype: :class:`CloudStackAffinityGroupType`
        """
        self.type = type_name

    def __repr__(self):
        return '<CloudStackAffinityGroupType: type=%s>' % self.type