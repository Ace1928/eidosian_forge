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
def _get_block_device_mapping_params(self, block_device_mapping):
    """
        Return a list of dictionaries with query parameters for
        a valid block device mapping.

        :param      mapping: List of dictionaries with the drive layout
        :type       mapping: ``list`` or ``dict``

        :return:    Dictionary representation of the drive mapping
        :rtype:     ``dict``
        """
    if not isinstance(block_device_mapping, (list, tuple)):
        raise AttributeError('block_device_mapping not list or tuple')
    params = {}
    for idx, mapping in enumerate(block_device_mapping):
        idx += 1
        if not isinstance(mapping, dict):
            raise AttributeError('mapping %s in block_device_mapping not a dict' % mapping)
        for k, v in mapping.items():
            if not isinstance(v, dict):
                params['BlockDeviceMapping.%d.%s' % (idx, k)] = str(v)
            else:
                for key, value in v.items():
                    params['BlockDeviceMapping.%d.%s.%s' % (idx, k, key)] = str(value)
    return params