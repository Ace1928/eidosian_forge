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
class NimbusNodeDriver(BaseEC2NodeDriver):
    """
    Driver class for Nimbus
    """
    type = Provider.NIMBUS
    name = 'Nimbus'
    website = 'http://www.nimbusproject.org/'
    country = 'Private'
    api_name = 'nimbus'
    region_name = 'nimbus'
    friendly_name = 'Nimbus Private Cloud'
    connectionCls = NimbusConnection
    signature_version = '2'

    def list_sizes(self, location=None):
        from libcloud.compute.constants.ec2_instance_types import INSTANCE_TYPES
        available_types = REGION_DETAILS_NIMBUS['instance_types']
        sizes = []
        for instance_type in available_types:
            attributes = INSTANCE_TYPES[instance_type]
            attributes = copy.deepcopy(attributes)
            attributes['price'] = None
            sizes.append(NodeSize(driver=self, **attributes))
        return sizes

    def ex_describe_addresses(self, nodes):
        """
        Nimbus doesn't support elastic IPs, so this is a pass-through.

        @inherits: :class:`EC2NodeDriver.ex_describe_addresses`
        """
        nodes_elastic_ip_mappings = {}
        for node in nodes:
            nodes_elastic_ip_mappings[node.id] = []
        return nodes_elastic_ip_mappings

    def ex_create_tags(self, resource, tags):
        """
        Nimbus doesn't support creating tags, so this is a pass-through.

        @inherits: :class:`EC2NodeDriver.ex_create_tags`
        """
        pass