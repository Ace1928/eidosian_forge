from tests.unit import unittest
from boto.ecs import ECSConnection
from tests.unit import AWSMockServiceTestCase
class TestECSConnection(AWSMockServiceTestCase):
    connection_class = ECSConnection

    def default_body(self):
        return b'\n            <Items>\n              <Request>\n              <IsValid>True</IsValid>\n              <ItemLookupRequest>\n                <ItemId>B00008OE6I</ItemId>\n              </ItemLookupRequest>\n              </Request>\n              <Item>\n                <ASIN>B00008OE6I</ASIN>\n                <ItemAttributes>\n                <Manufacturer>Canon</Manufacturer>\n                <ProductGroup>Photography</ProductGroup>\n                <Title>Canon PowerShot S400 4MP Digital Camera w/ 3x Optical Zoom</Title>\n               </ItemAttributes>\n              </Item>\n            </Items>\n        '

    def test_item_lookup(self):
        self.set_http_response(status_code=200)
        item_set = self.service_connection.item_lookup(ItemId='0316067938', ResponseGroup='Reviews')
        self.assert_request_parameters({'ItemId': '0316067938', 'Operation': 'ItemLookup', 'ResponseGroup': 'Reviews', 'Service': 'AWSECommerceService'}, ignore_params_values=['Version', 'AWSAccessKeyId', 'SignatureMethod', 'SignatureVersion', 'Timestamp'])
        items = list(item_set)
        self.assertEqual(len(items), 1)
        self.assertTrue(item_set.is_valid)
        self.assertEqual(items[0].ASIN, 'B00008OE6I')