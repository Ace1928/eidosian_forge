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
class EC2RouteTable:
    """
    Class which stores information about VPC Route Tables.

    Note: This class is VPC specific.
    """

    def __init__(self, id, name, routes, subnet_associations, propagating_gateway_ids, extra=None):
        """
        :param      id: The ID of the route table.
        :type       id: ``str``

        :param      name: The name of the route table.
        :type       name: ``str``

        :param      routes: A list of routes in the route table.
        :type       routes: ``list`` of :class:`EC2Route`

        :param      subnet_associations: A list of associations between the
                                         route table and one or more subnets.
        :type       subnet_associations: ``list`` of
                                         :class:`EC2SubnetAssociation`

        :param      propagating_gateway_ids: The list of IDs of any virtual
                                             private gateways propagating the
                                             routes.
        :type       propagating_gateway_ids: ``list``
        """
        self.id = id
        self.name = name
        self.routes = routes
        self.subnet_associations = subnet_associations
        self.propagating_gateway_ids = propagating_gateway_ids
        self.extra = extra or {}

    def __repr__(self):
        return '<EC2RouteTable: id=%s>' % self.id