from tests.unit import AWSMockServiceTestCase
from boto.ec2.connection import EC2Connection
from boto.ec2.reservedinstance import ReservedInstance
class TestReservedInstancesSet(AWSMockServiceTestCase):
    connection_class = EC2Connection

    def default_body(self):
        return b'\n<reservedInstancesSet>\n    <item>\n        <reservedInstancesId>ididididid</reservedInstancesId>\n        <instanceType>t1.micro</instanceType>\n        <start>2014-05-03T14:10:10.944Z</start>\n        <end>2014-05-03T14:10:11.000Z</end>\n        <duration>64800000</duration>\n        <fixedPrice>62.5</fixedPrice>\n        <usagePrice>0.0</usagePrice>\n        <instanceCount>5</instanceCount>\n        <productDescription>Linux/UNIX</productDescription>\n        <state>retired</state>\n        <instanceTenancy>default</instanceTenancy>\n        <currencyCode>USD</currencyCode>\n        <offeringType>Heavy Utilization</offeringType>\n        <recurringCharges>\n            <item>\n                <frequency>Hourly</frequency>\n                <amount>0.005</amount>\n            </item>\n        </recurringCharges>\n    </item>\n</reservedInstancesSet>'

    def test_get_all_reserved_instaces(self):
        self.set_http_response(status_code=200)
        response = self.service_connection.get_all_reserved_instances()
        self.assertEqual(len(response), 1)
        self.assertTrue(isinstance(response[0], ReservedInstance))
        self.assertEquals(response[0].id, 'ididididid')
        self.assertEquals(response[0].instance_count, 5)
        self.assertEquals(response[0].start, '2014-05-03T14:10:10.944Z')
        self.assertEquals(response[0].end, '2014-05-03T14:10:11.000Z')