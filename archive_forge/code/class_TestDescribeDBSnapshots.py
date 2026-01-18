from tests.unit import unittest
from tests.unit import AWSMockServiceTestCase
from boto.rds import RDSConnection
from boto.rds.dbsnapshot import DBSnapshot
from boto.rds import DBInstance
class TestDescribeDBSnapshots(AWSMockServiceTestCase):
    connection_class = RDSConnection

    def default_body(self):
        return '\n        <DescribeDBSnapshotsResponse xmlns="http://rds.amazonaws.com/doc/2013-05-15/">\n            <DescribeDBSnapshotsResult>\n                <DBSnapshots>\n                <DBSnapshot>\n                    <Port>3306</Port>\n                    <SnapshotCreateTime>2011-05-23T06:29:03.483Z</SnapshotCreateTime>\n                    <Engine>mysql</Engine>\n                    <Status>available</Status>\n                    <AvailabilityZone>us-east-1a</AvailabilityZone>\n                    <LicenseModel>general-public-license</LicenseModel>\n                    <InstanceCreateTime>2011-05-23T06:06:43.110Z</InstanceCreateTime>\n                    <AllocatedStorage>10</AllocatedStorage>\n                    <DBInstanceIdentifier>simcoprod01</DBInstanceIdentifier>\n                    <EngineVersion>5.1.50</EngineVersion>\n                    <DBSnapshotIdentifier>mydbsnapshot</DBSnapshotIdentifier>\n                    <SnapshotType>manual</SnapshotType>\n                    <MasterUsername>master</MasterUsername>\n                    <OptionGroupName>myoptiongroupname</OptionGroupName>\n                    <Iops>1000</Iops>\n                    <PercentProgress>100</PercentProgress>\n                    <SourceRegion>eu-west-1</SourceRegion>\n                    <VpcId>myvpc</VpcId>\n                </DBSnapshot>\n                <DBSnapshot>\n                    <Port>3306</Port>\n                    <SnapshotCreateTime>2011-03-11T07:20:24.082Z</SnapshotCreateTime>\n                    <Engine>mysql</Engine>\n                    <Status>available</Status>\n                    <AvailabilityZone>us-east-1a</AvailabilityZone>\n                    <LicenseModel>general-public-license</LicenseModel>\n                    <InstanceCreateTime>2010-08-04T23:27:36.420Z</InstanceCreateTime>\n                    <AllocatedStorage>50</AllocatedStorage>\n                    <DBInstanceIdentifier>mydbinstance</DBInstanceIdentifier>\n                    <EngineVersion>5.1.49</EngineVersion>\n                    <DBSnapshotIdentifier>mysnapshot1</DBSnapshotIdentifier>\n                    <SnapshotType>manual</SnapshotType>\n                    <MasterUsername>sa</MasterUsername>\n                    <OptionGroupName>myoptiongroupname</OptionGroupName>\n                    <Iops>1000</Iops>\n                </DBSnapshot>\n                <DBSnapshot>\n                    <Port>3306</Port>\n                    <SnapshotCreateTime>2012-04-02T00:01:24.082Z</SnapshotCreateTime>\n                    <Engine>mysql</Engine>\n                    <Status>available</Status>\n                    <AvailabilityZone>us-east-1d</AvailabilityZone>\n                    <LicenseModel>general-public-license</LicenseModel>\n                    <InstanceCreateTime>2010-07-16T00:06:59.107Z</InstanceCreateTime>\n                    <AllocatedStorage>60</AllocatedStorage>\n                    <DBInstanceIdentifier>simcoprod01</DBInstanceIdentifier>\n                    <EngineVersion>5.1.47</EngineVersion>\n                    <DBSnapshotIdentifier>rds:simcoprod01-2012-04-02-00-01</DBSnapshotIdentifier>\n                    <SnapshotType>automated</SnapshotType>\n                    <MasterUsername>master</MasterUsername>\n                    <OptionGroupName>myoptiongroupname</OptionGroupName>\n                    <Iops>1000</Iops>\n                </DBSnapshot>\n                </DBSnapshots>\n            </DescribeDBSnapshotsResult>\n            <ResponseMetadata>\n                <RequestId>c4191173-8506-11e0-90aa-eb648410240d</RequestId>\n            </ResponseMetadata>\n        </DescribeDBSnapshotsResponse>        \n        '

    def test_describe_dbinstances_by_instance(self):
        self.set_http_response(status_code=200)
        response = self.service_connection.get_all_dbsnapshots(instance_id='simcoprod01')
        self.assert_request_parameters({'Action': 'DescribeDBSnapshots', 'DBInstanceIdentifier': 'simcoprod01'}, ignore_params_values=['Version'])
        self.assertEqual(len(response), 3)
        self.assertIsInstance(response[0], DBSnapshot)
        self.assertEqual(response[0].id, 'mydbsnapshot')
        self.assertEqual(response[0].status, 'available')
        self.assertEqual(response[0].instance_id, 'simcoprod01')
        self.assertEqual(response[0].engine_version, '5.1.50')
        self.assertEqual(response[0].license_model, 'general-public-license')
        self.assertEqual(response[0].iops, 1000)
        self.assertEqual(response[0].option_group_name, 'myoptiongroupname')
        self.assertEqual(response[0].percent_progress, 100)
        self.assertEqual(response[0].snapshot_type, 'manual')
        self.assertEqual(response[0].source_region, 'eu-west-1')
        self.assertEqual(response[0].vpc_id, 'myvpc')