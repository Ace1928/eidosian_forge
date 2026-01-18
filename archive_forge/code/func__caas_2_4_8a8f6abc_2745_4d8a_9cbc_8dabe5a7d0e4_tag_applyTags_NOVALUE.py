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
def _caas_2_4_8a8f6abc_2745_4d8a_9cbc_8dabe5a7d0e4_tag_applyTags_NOVALUE(self, method, url, body, headers):
    request = ET.fromstring(body)
    if request.tag != '{urn:didata.com:api:cloud:types}applyTags':
        raise InvalidRequestError(request.tag)
    asset_type = findtext(request, 'assetType', TYPES_URN)
    asset_id = findtext(request, 'assetId', TYPES_URN)
    tag = request.find(fixxpath('tag', TYPES_URN))
    tag_key_name = findtext(tag, 'tagKeyName', TYPES_URN)
    value = findtext(tag, 'value', TYPES_URN)
    if asset_type is None:
        raise ValueError('assetType should not be empty')
    if asset_id is None:
        raise ValueError('assetId should not be empty')
    if tag_key_name is None:
        raise ValueError('tagKeyName should not be empty')
    if value is not None:
        raise ValueError('value should be empty')
    body = self.fixtures.load('tag_applyTags.xml')
    return (httplib.OK, body, {}, httplib.responses[httplib.OK])