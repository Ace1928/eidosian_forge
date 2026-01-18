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
def _oec_0_9_8a8f6abc_2745_4d8a_9cbc_8dabe5a7d0e4_server_11(self, method, url, body, headers):
    body = None
    action = url.split('?')[-1]
    if action == 'restart':
        body = self.fixtures.load('oec_0_9_8a8f6abc_2745_4d8a_9cbc_8dabe5a7d0e4_server_11_restart.xml')
    elif action == 'shutdown':
        body = self.fixtures.load('oec_0_9_8a8f6abc_2745_4d8a_9cbc_8dabe5a7d0e4_server_11_shutdown.xml')
    elif action == 'delete':
        body = self.fixtures.load('oec_0_9_8a8f6abc_2745_4d8a_9cbc_8dabe5a7d0e4_server_11_delete.xml')
    elif action == 'start':
        body = self.fixtures.load('oec_0_9_8a8f6abc_2745_4d8a_9cbc_8dabe5a7d0e4_server_11_start.xml')
    elif action == 'poweroff':
        body = self.fixtures.load('oec_0_9_8a8f6abc_2745_4d8a_9cbc_8dabe5a7d0e4_server_11_poweroff.xml')
    return (httplib.OK, body, {}, httplib.responses[httplib.OK])