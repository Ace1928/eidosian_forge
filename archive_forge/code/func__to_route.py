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
def _to_route(self, element):
    """
        Parse the XML element and return a route object

        :rtype:     :class: `EC2Route`
        """
    destination_cidr = findtext(element=element, xpath='destinationCidrBlock', namespace=NAMESPACE)
    gateway_id = findtext(element=element, xpath='gatewayId', namespace=NAMESPACE)
    instance_id = findtext(element=element, xpath='instanceId', namespace=NAMESPACE)
    owner_id = findtext(element=element, xpath='instanceOwnerId', namespace=NAMESPACE)
    interface_id = findtext(element=element, xpath='networkInterfaceId', namespace=NAMESPACE)
    state = findtext(element=element, xpath='state', namespace=NAMESPACE)
    origin = findtext(element=element, xpath='origin', namespace=NAMESPACE)
    vpc_peering_connection_id = findtext(element=element, xpath='vpcPeeringConnectionId', namespace=NAMESPACE)
    return EC2Route(destination_cidr, gateway_id, instance_id, owner_id, interface_id, state, origin, vpc_peering_connection_id)