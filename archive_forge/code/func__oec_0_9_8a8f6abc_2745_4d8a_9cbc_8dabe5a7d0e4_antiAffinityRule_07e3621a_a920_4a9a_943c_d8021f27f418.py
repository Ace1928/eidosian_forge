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
def _oec_0_9_8a8f6abc_2745_4d8a_9cbc_8dabe5a7d0e4_antiAffinityRule_07e3621a_a920_4a9a_943c_d8021f27f418(self, method, url, body, headers):
    body = self.fixtures.load('oec_0_9_8a8f6abc_2745_4d8a_9cbc_8dabe5a7d0e4_antiAffinityRule_delete.xml')
    return (httplib.OK, body, {}, httplib.responses[httplib.OK])