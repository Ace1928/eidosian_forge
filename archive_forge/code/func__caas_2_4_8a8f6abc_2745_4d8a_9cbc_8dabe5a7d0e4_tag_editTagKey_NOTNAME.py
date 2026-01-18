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
def _caas_2_4_8a8f6abc_2745_4d8a_9cbc_8dabe5a7d0e4_tag_editTagKey_NOTNAME(self, method, url, body, headers):
    request = ET.fromstring(body)
    if request.tag != '{urn:didata.com:api:cloud:types}editTagKey':
        raise InvalidRequestError(request.tag)
    name = findtext(request, 'name', TYPES_URN)
    description = findtext(request, 'description', TYPES_URN)
    value_required = findtext(request, 'valueRequired', TYPES_URN)
    display_on_report = findtext(request, 'displayOnReport', TYPES_URN)
    if name is not None:
        raise ValueError('Name should be empty')
    if description is None:
        raise ValueError('Description should not be empty')
    if value_required is None:
        raise ValueError('valueRequired should not be empty')
    if display_on_report is None:
        raise ValueError('displayOnReport should not be empty')
    body = self.fixtures.load('tag_editTagKey.xml')
    return (httplib.OK, body, {}, httplib.responses[httplib.OK])