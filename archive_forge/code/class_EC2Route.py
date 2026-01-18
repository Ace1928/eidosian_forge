import re
import copy
import time
import base64
import warnings
from typing import List
from libcloud.pricing import get_size_price
from libcloud.utils.py3 import ET, b, basestring, ensure_string
from libcloud.utils.xml import findall, findattr, findtext, fixxpath
from libcloud.common.aws import DEFAULT_SIGNATURE_VERSION, AWSBaseResponse, SignedAWSConnection
from libcloud.common.types import LibcloudError, InvalidCredsError, MalformedResponseError
from libcloud.compute.base import (
from libcloud.compute.types import (
from libcloud.utils.iso8601 import parse_date, parse_date_allow_empty
from libcloud.utils.publickey import get_pubkey_comment, get_pubkey_ssh2_fingerprint
from libcloud.compute.providers import Provider
from libcloud.compute.constants.ec2_region_details_partial import (
class EC2Route:
    """
    Class which stores information about a Route.

    Note: This class is VPC specific.
    """

    def __init__(self, cidr, gateway_id, instance_id, owner_id, interface_id, state, origin, vpc_peering_connection_id):
        """
        :param      cidr: The CIDR block used for the destination match.
        :type       cidr: ``str``

        :param      gateway_id: The ID of a gateway attached to the VPC.
        :type       gateway_id: ``str``

        :param      instance_id: The ID of a NAT instance in the VPC.
        :type       instance_id: ``str``

        :param      owner_id: The AWS account ID of the owner of the instance.
        :type       owner_id: ``str``

        :param      interface_id: The ID of the network interface.
        :type       interface_id: ``str``

        :param      state: The state of the route (active | blackhole).
        :type       state: ``str``

        :param      origin: Describes how the route was created.
        :type       origin: ``str``

        :param      vpc_peering_connection_id: The ID of the VPC
                                               peering connection.
        :type       vpc_peering_connection_id: ``str``
        """
        self.cidr = cidr
        self.gateway_id = gateway_id
        self.instance_id = instance_id
        self.owner_id = owner_id
        self.interface_id = interface_id
        self.state = state
        self.origin = origin
        self.vpc_peering_connection_id = vpc_peering_connection_id

    def __repr__(self):
        return '<EC2Route: cidr=%s>' % self.cidr