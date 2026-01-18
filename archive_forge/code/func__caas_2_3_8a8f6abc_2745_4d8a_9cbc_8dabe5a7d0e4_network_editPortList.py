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
def _caas_2_3_8a8f6abc_2745_4d8a_9cbc_8dabe5a7d0e4_network_editPortList(self, method, url, body, headers):
    request = ET.fromstring(body)
    if request.tag != '{urn:didata.com:api:cloud:types}editPortList':
        raise InvalidRequestError(request.tag)
    ports_required = findall(request, 'port', TYPES_URN)
    child_port_list_required = findall(request, 'childPortListId', TYPES_URN)
    if 0 == len(ports_required) and 0 == len(child_port_list_required):
        raise ValueError('At least one port element or one childPortListId element must be provided')
    if ports_required[0].get('begin') is None:
        raise ValueError('PORT begin value should not be empty')
    body = self.fixtures.load('port_list_edit.xml')
    return (httplib.OK, body, {}, httplib.responses[httplib.OK])