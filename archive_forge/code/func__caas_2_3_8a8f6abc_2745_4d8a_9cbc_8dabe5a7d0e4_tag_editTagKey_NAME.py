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
def _caas_2_3_8a8f6abc_2745_4d8a_9cbc_8dabe5a7d0e4_tag_editTagKey_NAME(self, method, url, body, headers):
    request = ET.fromstring(body)
    if request.tag != '{urn:didata.com:api:cloud:types}editTagKey':
        raise InvalidRequestError(request.tag)
    name = findtext(request, 'name', TYPES_URN)
    description = findtext(request, 'description', TYPES_URN)
    value_required = findtext(request, 'valueRequired', TYPES_URN)
    display_on_report = findtext(request, 'displayOnReport', TYPES_URN)
    if name is None:
        raise ValueError('Name must have a value in the request')
    if description is not None:
        raise ValueError('Description should be empty')
    if value_required is not None:
        raise ValueError('valueRequired should be empty')
    if display_on_report is not None:
        raise ValueError('displayOnReport should be empty')
    body = self.fixtures.load('tag_editTagKey.xml')
    return (httplib.OK, body, {}, httplib.responses[httplib.OK])