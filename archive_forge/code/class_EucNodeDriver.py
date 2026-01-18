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
class EucNodeDriver(BaseEC2NodeDriver):
    """
    Driver class for Eucalyptus
    """
    name = 'Eucalyptus'
    website = 'http://www.eucalyptus.com/'
    api_name = 'ec2_us_east'
    region_name = 'us-east-1'
    connectionCls = EucConnection
    signature_version = '2'

    def __init__(self, key, secret=None, secure=True, host=None, path=None, port=None, api_version=DEFAULT_EUCA_API_VERSION):
        """
        @inherits: :class:`EC2NodeDriver.__init__`

        :param    path: The host where the API can be reached.
        :type     path: ``str``

        :param    api_version: The API version to extend support for
                               Eucalyptus proprietary API calls
        :type     api_version: ``str``
        """
        super().__init__(key, secret, secure, host, port)
        if path is None:
            path = '/services/Eucalyptus'
        self.path = path
        self.EUCA_NAMESPACE = 'http://msgs.eucalyptus.com/%s' % api_version

    def list_locations(self):
        raise NotImplementedError('list_locations not implemented for this driver')

    def _to_sizes(self, response):
        return [self._to_size(el) for el in response.findall(fixxpath(xpath='instanceTypeDetails/item', namespace=self.EUCA_NAMESPACE))]

    def _to_size(self, el):
        name = findtext(element=el, xpath='name', namespace=self.EUCA_NAMESPACE)
        cpu = findtext(element=el, xpath='cpu', namespace=self.EUCA_NAMESPACE)
        disk = findtext(element=el, xpath='disk', namespace=self.EUCA_NAMESPACE)
        memory = findtext(element=el, xpath='memory', namespace=self.EUCA_NAMESPACE)
        return NodeSize(id=name, name=name, ram=int(memory), disk=int(disk), bandwidth=None, price=None, driver=EucNodeDriver, extra={'cpu': int(cpu)})

    def list_sizes(self):
        """
        Lists available nodes sizes.

        :rtype: ``list`` of :class:`NodeSize`
        """
        params = {'Action': 'DescribeInstanceTypes'}
        response = self.connection.request(self.path, params=params).object
        return self._to_sizes(response)

    def _add_instance_filter(self, params, node):
        """
        Eucalyptus driver doesn't support filtering on instance id so this is a
        no-op.
        """
        pass