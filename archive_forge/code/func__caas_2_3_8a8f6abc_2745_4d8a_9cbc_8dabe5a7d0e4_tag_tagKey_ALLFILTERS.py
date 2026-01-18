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
def _caas_2_3_8a8f6abc_2745_4d8a_9cbc_8dabe5a7d0e4_tag_tagKey_ALLFILTERS(self, method, url, body, headers):
    _, params = url.split('?')
    parameters = params.split('&')
    for parameter in parameters:
        key, value = parameter.split('=')
        if key == 'id':
            assert value == 'fake_id'
        elif key == 'name':
            assert value == 'fake_name'
        elif key == 'valueRequired':
            assert value == 'false'
        elif key == 'displayOnReport':
            assert value == 'false'
        elif key == 'pageSize':
            assert value == '250'
        else:
            raise ValueError('Could not find in url parameters {}:{}'.format(key, value))
    body = self.fixtures.load('tag_tagKey_list.xml')
    return (httplib.OK, body, {}, httplib.responses[httplib.OK])