import re
import copy
import time
import base64
import hashlib
from libcloud.utils.py3 import b, httplib
from libcloud.utils.misc import dict2str, str2list, str2dicts, get_secure_random_string
from libcloud.common.base import Response, JsonResponse, ConnectionUserAndKey
from libcloud.common.types import ProviderError, InvalidCredsError
from libcloud.compute.base import Node, KeyPair, NodeSize, NodeImage, NodeDriver, is_private_subnet
from libcloud.compute.types import Provider, NodeState
from libcloud.utils.iso8601 import parse_date
from libcloud.common.cloudsigma import (
class CloudSigmaFirewallPolicy:
    """
    Represents a CloudSigma firewall policy.
    """

    def __init__(self, id, name, rules):
        """
        :param id: Policy ID.
        :type id: ``str``

        :param name: Policy name.
        :type name: ``str``

        :param rules: Rules associated with this policy.
        :type rules: ``list`` of :class:`.CloudSigmaFirewallPolicyRule` objects
        """
        self.id = id
        self.name = name
        self.rules = rules if rules else []

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return '<CloudSigmaFirewallPolicy id=%s, name=%s rules=%s>' % (self.id, self.name, repr(self.rules))