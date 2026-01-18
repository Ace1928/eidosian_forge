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
class ElasticIP:
    """
    Represents information about an elastic IP address

    :param      ip: The elastic IP address
    :type       ip: ``str``

    :param      domain: The domain that the IP resides in (EC2-Classic/VPC).
                        EC2 classic is represented with standard and VPC
                        is represented with vpc.
    :type       domain: ``str``

    :param      instance_id: The identifier of the instance which currently
                             has the IP associated.
    :type       instance_id: ``str``

    Note: This class is used to support both EC2 and VPC IPs.
          For VPC specific attributes are stored in the extra
          dict to make promotion to the base API easier.
    """

    def __init__(self, ip, domain, instance_id, extra=None):
        self.ip = ip
        self.domain = domain
        self.instance_id = instance_id
        self.extra = extra or {}

    def __repr__(self):
        return '<ElasticIP: ip=%s, domain=%s, instance_id=%s>' % (self.ip, self.domain, self.instance_id)