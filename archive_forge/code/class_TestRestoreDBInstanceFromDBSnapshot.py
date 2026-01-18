from tests.unit import unittest
from tests.unit import AWSMockServiceTestCase
from boto.rds import RDSConnection
from boto.rds.dbsnapshot import DBSnapshot
from boto.rds import DBInstance
class TestRestoreDBInstanceFromDBSnapshot(AWSMockServiceTestCase):
    connection_class = RDSConnection

    def default_body(self):
        return '\n        <RestoreDBInstanceFromDBSnapshotResponse xmlns="http://rds.amazonaws.com/doc/2013-05-15/">\n            <RestoreDBInstanceFromDBSnapshotResult>\n                <DBInstance>\n                <ReadReplicaDBInstanceIdentifiers/>\n                <Engine>mysql</Engine>\n                <PendingModifiedValues/>\n                <BackupRetentionPeriod>1</BackupRetentionPeriod>\n                <MultiAZ>false</MultiAZ>\n                <LicenseModel>general-public-license</LicenseModel>\n                <DBInstanceStatus>creating</DBInstanceStatus>\n                <EngineVersion>5.1.50</EngineVersion>\n                <DBInstanceIdentifier>myrestoreddbinstance</DBInstanceIdentifier>\n                <DBParameterGroups>\n                    <DBParameterGroup>\n                    <ParameterApplyStatus>in-sync</ParameterApplyStatus>\n                    <DBParameterGroupName>default.mysql5.1</DBParameterGroupName>\n                    </DBParameterGroup>\n                </DBParameterGroups>\n                <DBSecurityGroups>\n                    <DBSecurityGroup>\n                    <Status>active</Status>\n                    <DBSecurityGroupName>default</DBSecurityGroupName>\n                    </DBSecurityGroup>\n                </DBSecurityGroups>\n                <PreferredBackupWindow>00:00-00:30</PreferredBackupWindow>\n                <AutoMinorVersionUpgrade>true</AutoMinorVersionUpgrade>\n                <PreferredMaintenanceWindow>sat:07:30-sat:08:00</PreferredMaintenanceWindow>\n                <AllocatedStorage>10</AllocatedStorage>\n                <DBInstanceClass>db.m1.large</DBInstanceClass>\n                <MasterUsername>master</MasterUsername>\n                </DBInstance>\n            </RestoreDBInstanceFromDBSnapshotResult>\n            <ResponseMetadata>\n                <RequestId>7ca622e8-8508-11e0-bd9b-a7b1ece36d51</RequestId>\n            </ResponseMetadata>\n        </RestoreDBInstanceFromDBSnapshotResponse>\n        '

    def test_restore_dbinstance_from_dbsnapshot(self):
        self.set_http_response(status_code=200)
        response = self.service_connection.restore_dbinstance_from_dbsnapshot('mydbsnapshot', 'myrestoreddbinstance', 'db.m1.large', '3306', 'us-east-1a', 'false', 'true')
        self.assert_request_parameters({'Action': 'RestoreDBInstanceFromDBSnapshot', 'DBSnapshotIdentifier': 'mydbsnapshot', 'DBInstanceIdentifier': 'myrestoreddbinstance', 'DBInstanceClass': 'db.m1.large', 'Port': '3306', 'AvailabilityZone': 'us-east-1a', 'MultiAZ': 'false', 'AutoMinorVersionUpgrade': 'true'}, ignore_params_values=['Version'])
        self.assertIsInstance(response, DBInstance)
        self.assertEqual(response.id, 'myrestoreddbinstance')
        self.assertEqual(response.status, 'creating')
        self.assertEqual(response.instance_class, 'db.m1.large')
        self.assertEqual(response.multi_az, False)