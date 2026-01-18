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
def _get_common_security_group_params(self, group_id, protocol, from_port, to_port, cidr_ips, group_pairs, description=None):
    """
        Return a dictionary with common query parameters which are used when
        operating on security groups.

        :rtype: ``dict``
        """
    params = {'GroupId': group_id, 'IpPermissions.1.IpProtocol': protocol, 'IpPermissions.1.FromPort': from_port, 'IpPermissions.1.ToPort': to_port}
    if cidr_ips is not None:
        ip_ranges = {}
        for index, cidr_ip in enumerate(cidr_ips):
            index += 1
            ip_ranges['IpPermissions.1.IpRanges.%s.CidrIp' % index] = cidr_ip
            if description is not None:
                ip_ranges['IpPermissions.1.IpRanges.%s.Description' % index] = description
        params.update(ip_ranges)
    if group_pairs is not None:
        user_groups = {}
        for index, group_pair in enumerate(group_pairs):
            index += 1
            if 'group_id' in group_pair.keys():
                user_groups['IpPermissions.1.Groups.%s.GroupId' % index] = group_pair['group_id']
            if 'group_name' in group_pair.keys():
                user_groups['IpPermissions.1.Groups.%s.GroupName' % index] = group_pair['group_name']
            if 'user_id' in group_pair.keys():
                user_groups['IpPermissions.1.Groups.%s.UserId' % index] = group_pair['user_id']
        params.update(user_groups)
    return params