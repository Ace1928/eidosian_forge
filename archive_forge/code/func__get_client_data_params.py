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
def _get_client_data_params(self, client_data):
    """
        Return a dictionary with query parameters for
        a valid client data.

        :param      client_data: List of dictionaries with the disk
                                 upload details
        :type       client_data: ``dict``

        :return:    Dictionary representation of the client data
        :rtype:     ``dict``
        """
    if not isinstance(client_data, (list, tuple)):
        raise AttributeError('client_data not list or tuple')
    params = {}
    for idx, content in enumerate(client_data):
        idx += 1
        if not isinstance(content, dict):
            raise AttributeError('content %s in client_datanot a dict' % content)
        for k, v in content.items():
            params['ClientData.%s' % k] = str(v)
    return params