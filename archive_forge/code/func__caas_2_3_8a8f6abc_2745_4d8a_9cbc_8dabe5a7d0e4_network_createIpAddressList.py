import sys
from types import GeneratorType
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import ET, httplib
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.common.types import InvalidCredsError
from libcloud.compute.base import Node, NodeLocation, NodeAuthPassword
from libcloud.test.compute import TestCaseMixin
from libcloud.test.secrets import DIMENSIONDATA_PARAMS
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.common.dimensiondata import (
from libcloud.compute.drivers.dimensiondata import DimensionDataNic
from libcloud.compute.drivers.dimensiondata import DimensionDataNodeDriver as DimensionData
def _caas_2_3_8a8f6abc_2745_4d8a_9cbc_8dabe5a7d0e4_network_createIpAddressList(self, method, url, body, headers):
    request = ET.fromstring(body)
    if request.tag != '{urn:didata.com:api:cloud:types}createIpAddressList':
        raise InvalidRequestError(request.tag)
    net_domain = findtext(request, 'networkDomainId', TYPES_URN)
    if net_domain is None:
        raise ValueError('Network Domain should not be empty')
    name = findtext(request, 'name', TYPES_URN)
    if name is None:
        raise ValueError('Name should not be empty')
    ip_version = findtext(request, 'ipVersion', TYPES_URN)
    if ip_version is None:
        raise ValueError('IP Version should not be empty')
    ip_address_col_required = findall(request, 'ipAddress', TYPES_URN)
    child_ip_address_required = findall(request, 'childIpAddressListId', TYPES_URN)
    if 0 == len(ip_address_col_required) and 0 == len(child_ip_address_required):
        raise ValueError('At least one ipAddress element or one childIpAddressListId element must be provided.')
    if ip_address_col_required[0].get('begin') is None:
        raise ValueError('IP Address should not be empty')
    body = self.fixtures.load('ip_address_list_create.xml')
    return (httplib.OK, body, {}, httplib.responses[httplib.OK])