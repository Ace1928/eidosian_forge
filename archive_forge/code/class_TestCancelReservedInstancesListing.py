from datetime import datetime, timedelta
from mock import MagicMock, Mock
from tests.unit import unittest
from tests.unit import AWSMockServiceTestCase
import boto.ec2
from boto.regioninfo import RegionInfo
from boto.ec2.blockdevicemapping import BlockDeviceType, BlockDeviceMapping
from boto.ec2.connection import EC2Connection
from boto.ec2.snapshot import Snapshot
from boto.ec2.reservedinstance import ReservedInstancesConfiguration
from boto.compat import http_client
class TestCancelReservedInstancesListing(TestEC2ConnectionBase):

    def default_body(self):
        return b'\n            <CancelReservedInstancesListingResponse>\n                <requestId>request_id</requestId>\n                <reservedInstancesListingsSet>\n                    <item>\n                        <reservedInstancesListingId>listing_id</reservedInstancesListingId>\n                        <reservedInstancesId>instance_id</reservedInstancesId>\n                        <createDate>2012-07-12T16:55:28.000Z</createDate>\n                        <updateDate>2012-07-12T16:55:28.000Z</updateDate>\n                        <status>cancelled</status>\n                        <statusMessage>CANCELLED</statusMessage>\n                        <instanceCounts>\n                            <item>\n                                <state>Available</state>\n                                <instanceCount>0</instanceCount>\n                            </item>\n                            <item>\n                                <state>Sold</state>\n                                <instanceCount>0</instanceCount>\n                            </item>\n                            <item>\n                                <state>Cancelled</state>\n                                <instanceCount>1</instanceCount>\n                            </item>\n                            <item>\n                                <state>Pending</state>\n                                <instanceCount>0</instanceCount>\n                            </item>\n                        </instanceCounts>\n                        <priceSchedules>\n                            <item>\n                                <term>5</term>\n                                <price>166.64</price>\n                                <currencyCode>USD</currencyCode>\n                                <active>false</active>\n                            </item>\n                            <item>\n                                <term>4</term>\n                                <price>133.32</price>\n                                <currencyCode>USD</currencyCode>\n                                <active>false</active>\n                            </item>\n                            <item>\n                                <term>3</term>\n                                <price>99.99</price>\n                                <currencyCode>USD</currencyCode>\n                                <active>false</active>\n                            </item>\n                            <item>\n                                <term>2</term>\n                                <price>66.66</price>\n                                <currencyCode>USD</currencyCode>\n                                <active>false</active>\n                            </item>\n                            <item>\n                                <term>1</term>\n                                <price>33.33</price>\n                                <currencyCode>USD</currencyCode>\n                                <active>false</active>\n                            </item>\n                        </priceSchedules>\n                        <tagSet/>\n                        <clientToken>XqJIt1342112125076</clientToken>\n                    </item>\n                </reservedInstancesListingsSet>\n            </CancelReservedInstancesListingResponse>\n        '

    def test_reserved_instances_listing(self):
        self.set_http_response(status_code=200)
        response = self.ec2.cancel_reserved_instances_listing()
        self.assertEqual(len(response), 1)
        cancellation = response[0]
        self.assertEqual(cancellation.status, 'cancelled')
        self.assertEqual(cancellation.status_message, 'CANCELLED')
        self.assertEqual(len(cancellation.instance_counts), 4)
        first = cancellation.instance_counts[0]
        self.assertEqual(first.state, 'Available')
        self.assertEqual(first.instance_count, 0)
        self.assertEqual(len(cancellation.price_schedules), 5)
        schedule = cancellation.price_schedules[0]
        self.assertEqual(schedule.term, 5)
        self.assertEqual(schedule.price, '166.64')
        self.assertEqual(schedule.currency_code, 'USD')
        self.assertEqual(schedule.active, False)