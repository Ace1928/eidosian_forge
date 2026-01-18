from tests.unit import unittest
from tests.unit import AWSMockServiceTestCase
from boto.rds import RDSConnection
from boto.rds.dbsnapshot import DBSnapshot
from boto.rds import DBInstance
class TestCreateDBSnapshot(AWSMockServiceTestCase):
    connection_class = RDSConnection

    def default_body(self):
        return '\n        <CreateDBSnapshotResponse xmlns="http://rds.amazonaws.com/doc/2013-05-15/">\n            <CreateDBSnapshotResult>\n                <DBSnapshot>\n                <Port>3306</Port>\n                <Engine>mysql</Engine>\n                <Status>creating</Status>\n                <AvailabilityZone>us-east-1a</AvailabilityZone>\n                <LicenseModel>general-public-license</LicenseModel>\n                <InstanceCreateTime>2011-05-23T06:06:43.110Z</InstanceCreateTime>\n                <AllocatedStorage>10</AllocatedStorage>\n                <DBInstanceIdentifier>simcoprod01</DBInstanceIdentifier>\n                <EngineVersion>5.1.50</EngineVersion>\n                <DBSnapshotIdentifier>mydbsnapshot</DBSnapshotIdentifier>\n                <SnapshotType>manual</SnapshotType>\n                <MasterUsername>master</MasterUsername>\n                </DBSnapshot>\n            </CreateDBSnapshotResult>\n            <ResponseMetadata>\n                <RequestId>c4181d1d-8505-11e0-90aa-eb648410240d</RequestId>\n            </ResponseMetadata>\n        </CreateDBSnapshotResponse>\n        '

    def test_create_dbinstance(self):
        self.set_http_response(status_code=200)
        response = self.service_connection.create_dbsnapshot('mydbsnapshot', 'simcoprod01')
        self.assert_request_parameters({'Action': 'CreateDBSnapshot', 'DBSnapshotIdentifier': 'mydbsnapshot', 'DBInstanceIdentifier': 'simcoprod01'}, ignore_params_values=['Version'])
        self.assertIsInstance(response, DBSnapshot)
        self.assertEqual(response.id, 'mydbsnapshot')
        self.assertEqual(response.instance_id, 'simcoprod01')
        self.assertEqual(response.status, 'creating')