import sys
from types import GeneratorType
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import ET, httplib
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.common.types import InvalidCredsError
from libcloud.compute.base import Node, NodeLocation, NodeAuthPassword
from libcloud.test.secrets import DIMENSIONDATA_PARAMS
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.common.dimensiondata import (
from libcloud.compute.drivers.dimensiondata import DimensionDataNic
from libcloud.compute.drivers.dimensiondata import DimensionDataNodeDriver as DimensionData
def _caas_2_4_8a8f6abc_2745_4d8a_9cbc_8dabe5a7d0e4_network_publicIpBlock(self, method, url, body, headers):
    body = self.fixtures.load('network_publicIpBlock.xml')
    return (httplib.OK, body, {}, httplib.responses[httplib.OK])