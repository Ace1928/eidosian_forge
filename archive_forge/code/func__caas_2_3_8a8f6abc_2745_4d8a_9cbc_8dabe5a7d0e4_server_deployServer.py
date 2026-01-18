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
def _caas_2_3_8a8f6abc_2745_4d8a_9cbc_8dabe5a7d0e4_server_deployServer(self, method, url, body, headers):
    request = ET.fromstring(body)
    if request.tag != '{urn:didata.com:api:cloud:types}deployServer':
        raise InvalidRequestError(request.tag)
    network = request.find(fixxpath('network', TYPES_URN))
    network_info = request.find(fixxpath('networkInfo', TYPES_URN))
    if network is not None:
        if network_info is not None:
            raise InvalidRequestError('Request has both MCP1 and MCP2 values')
        ipv4 = findtext(network, 'privateIpv4', TYPES_URN)
        networkId = findtext(network, 'networkId', TYPES_URN)
        if ipv4 is None and networkId is None:
            raise InvalidRequestError('Invalid request MCP1 requests need privateIpv4 or networkId')
    elif network_info is not None:
        if network is not None:
            raise InvalidRequestError('Request has both MCP1 and MCP2 values')
        primary_nic = network_info.find(fixxpath('primaryNic', TYPES_URN))
        ipv4 = findtext(primary_nic, 'privateIpv4', TYPES_URN)
        vlanId = findtext(primary_nic, 'vlanId', TYPES_URN)
        if ipv4 is None and vlanId is None:
            raise InvalidRequestError('Invalid request MCP2 requests need privateIpv4 or vlanId')
    else:
        raise InvalidRequestError('Invalid request, does not have network or network_info in XML')
    body = self.fixtures.load('server_deployServer.xml')
    return (httplib.OK, body, {}, httplib.responses[httplib.OK])