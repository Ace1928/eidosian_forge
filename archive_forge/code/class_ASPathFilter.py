import abc
from abc import ABCMeta
from abc import abstractmethod
from copy import copy
import logging
import functools
import netaddr
from os_ken.lib.packet.bgp import RF_IPv4_UC
from os_ken.lib.packet.bgp import RouteTargetMembershipNLRI
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_EXTENDED_COMMUNITIES
from os_ken.lib.packet.bgp import BGPPathAttributeLocalPref
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_AS_PATH
from os_ken.services.protocols.bgp.base import OrderedDict
from os_ken.services.protocols.bgp.constants import VPN_TABLE
from os_ken.services.protocols.bgp.constants import VRF_TABLE
from os_ken.services.protocols.bgp.model import OutgoingRoute
from os_ken.services.protocols.bgp.processor import BPR_ONLY_PATH
from os_ken.services.protocols.bgp.processor import BPR_UNKNOWN
@functools.total_ordering
class ASPathFilter(Filter):
    """
    Used to specify a prefix for AS_PATH attribute.

    We can create ASPathFilter object as follows::

        as_path_filter = ASPathFilter(65000,policy=ASPathFilter.TOP)

    ================ ==================================================
    Attribute        Description
    ================ ==================================================
    as_number        A AS number used for this filter
    policy           One of the following values.

                     | ASPathFilter.POLICY_TOP
                     | ASPathFilter.POLICY_END
                     | ASPathFilter.POLICY_INCLUDE
                     | ASPathFilter.POLICY_NOT_INCLUDE
    ================ ==================================================

    Meaning of each policy is as follows:

    ================== ==================================================
    Policy             Description
    ================== ==================================================
    POLICY_TOP         Filter checks if the specified AS number
                       is at the top of AS_PATH attribute.
    POLICY_END         Filter checks is the specified AS number
                       is at the last of AS_PATH attribute.
    POLICY_INCLUDE     Filter checks if specified AS number exists
                       in AS_PATH attribute.
    POLICY_NOT_INCLUDE Opposite to POLICY_INCLUDE.
    ================== ==================================================
    """
    POLICY_TOP = 2
    POLICY_END = 3
    POLICY_INCLUDE = 4
    POLICY_NOT_INCLUDE = 5

    def __init__(self, as_number, policy):
        super(ASPathFilter, self).__init__(policy)
        self._as_number = as_number

    def __lt__(self, other):
        return self.as_number < other.as_number

    def __eq__(self, other):
        return self.as_number == other.as_number

    def __repr__(self):
        policy = 'TOP'
        if self._policy == self.POLICY_INCLUDE:
            policy = 'INCLUDE'
        elif self._policy == self.POLICY_NOT_INCLUDE:
            policy = 'NOT_INCLUDE'
        elif self._policy == self.POLICY_END:
            policy = 'END'
        return 'ASPathFilter(as_number=%s,policy=%s)' % (self._as_number, policy)

    @property
    def as_number(self):
        return self._as_number

    @property
    def policy(self):
        return self._policy

    def evaluate(self, path):
        """ This method evaluates as_path list.

        Returns this object's policy and the result of matching.
        If the specified AS number matches this object's AS number
        according to the policy,
        this method returns True as the matching result.

        ``path`` specifies the path.
        """
        path_aspath = path.pathattr_map.get(BGP_ATTR_TYPE_AS_PATH)
        path_seg_list = path_aspath.path_seg_list
        if path_seg_list:
            path_seg = path_seg_list[0]
        else:
            path_seg = []
        result = False
        LOG.debug('path_seg : %s', path_seg)
        if self.policy == ASPathFilter.POLICY_TOP:
            if len(path_seg) > 0 and path_seg[0] == self._as_number:
                result = True
        elif self.policy == ASPathFilter.POLICY_INCLUDE:
            for aspath in path_seg:
                LOG.debug('POLICY_INCLUDE as_number : %s', aspath)
                if aspath == self._as_number:
                    result = True
                    break
        elif self.policy == ASPathFilter.POLICY_END:
            if len(path_seg) > 0 and path_seg[-1] == self._as_number:
                result = True
        elif self.policy == ASPathFilter.POLICY_NOT_INCLUDE:
            if self._as_number not in path_seg:
                result = True
        return (self.policy, result)

    def clone(self):
        """ This method clones ASPathFilter object.

        Returns ASPathFilter object that has the same values with the
        original one.
        """
        return self.__class__(self._as_number, policy=self._policy)