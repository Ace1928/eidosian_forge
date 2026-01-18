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
def _oec_0_9_8a8f6abc_2745_4d8a_9cbc_8dabe5a7d0e4_server_e75ead52_692f_4314_8725_c8a4f4d13a87(self, method, url, body, headers):
    if method == 'GET':
        body = self.fixtures.load('oec_0_9_8a8f6abc_2745_4d8a_9cbc_8dabe5a7d0e4_server_e75ead52_692f_4314_8725_c8a4f4d13a87.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])
    if method == 'POST':
        body = self.fixtures.load('oec_0_9_8a8f6abc_2745_4d8a_9cbc_8dabe5a7d0e4_server_e75ead52_692f_4314_8725_c8a4f4d13a87_POST.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])