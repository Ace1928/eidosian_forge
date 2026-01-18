from tests.unit import unittest
from tests.unit import AWSMockServiceTestCase
from boto.ec2.securitygroup import SecurityGroup
from boto.rds import RDSConnection
from boto.rds.vpcsecuritygroupmembership import VPCSecurityGroupMembership
from boto.rds.parametergroup import ParameterGroup
from boto.rds.logfile import LogFile, LogFileObject
import xml.sax.saxutils as saxutils
class TestRDSCCreateDBInstance(AWSMockServiceTestCase):
    connection_class = RDSConnection

    def setUp(self):
        super(TestRDSCCreateDBInstance, self).setUp()

    def default_body(self):
        return '\n        <CreateDBInstanceResponse xmlns="http://rds.amazonaws.com/doc/2013-05-15/">\n            <CreateDBInstanceResult>\n                <DBInstance>\n                    <ReadReplicaDBInstanceIdentifiers/>\n                    <Engine>mysql</Engine>\n                    <PendingModifiedValues>\n                        <MasterUserPassword>****</MasterUserPassword>\n                    </PendingModifiedValues>\n                    <BackupRetentionPeriod>0</BackupRetentionPeriod>\n                    <MultiAZ>false</MultiAZ>\n                    <LicenseModel>general-public-license</LicenseModel>\n                    <DBSubnetGroup>\n                        <VpcId>990524496922</VpcId>\n                        <SubnetGroupStatus>Complete</SubnetGroupStatus>\n                        <DBSubnetGroupDescription>description</DBSubnetGroupDescription>\n                        <DBSubnetGroupName>subnet_grp1</DBSubnetGroupName>\n                        <Subnets>\n                            <Subnet>\n                                <SubnetStatus>Active</SubnetStatus>\n                                <SubnetIdentifier>subnet-7c5b4115</SubnetIdentifier>\n                                <SubnetAvailabilityZone>\n                                    <Name>us-east-1c</Name>\n                                </SubnetAvailabilityZone>\n                            </Subnet>\n                            <Subnet>\n                                <SubnetStatus>Active</SubnetStatus>\n                                <SubnetIdentifier>subnet-7b5b4112</SubnetIdentifier>\n                                <SubnetAvailabilityZone>\n                                    <Name>us-east-1b</Name>\n                                </SubnetAvailabilityZone>\n                            </Subnet>\n                            <Subnet>\n                                <SubnetStatus>Active</SubnetStatus>\n                                <SubnetIdentifier>subnet-3ea6bd57</SubnetIdentifier>\n                                <SubnetAvailabilityZone>\n                                    <Name>us-east-1d</Name>\n                                </SubnetAvailabilityZone>\n                            </Subnet>\n                        </Subnets>\n                    </DBSubnetGroup>\n                    <DBInstanceStatus>creating</DBInstanceStatus>\n                    <EngineVersion>5.1.50</EngineVersion>\n                    <DBInstanceIdentifier>simcoprod01</DBInstanceIdentifier>\n                    <DBParameterGroups>\n                        <DBParameterGroup>\n                            <ParameterApplyStatus>in-sync</ParameterApplyStatus>\n                            <DBParameterGroupName>default.mysql5.1</DBParameterGroupName>\n                        </DBParameterGroup>\n                    </DBParameterGroups>\n                    <DBSecurityGroups>\n                        <DBSecurityGroup>\n                            <Status>active</Status>\n                            <DBSecurityGroupName>default</DBSecurityGroupName>\n                        </DBSecurityGroup>\n                    </DBSecurityGroups>\n                    <PreferredBackupWindow>00:00-00:30</PreferredBackupWindow>\n                    <AutoMinorVersionUpgrade>true</AutoMinorVersionUpgrade>\n                    <PreferredMaintenanceWindow>sat:07:30-sat:08:00</PreferredMaintenanceWindow>\n                        <AllocatedStorage>10</AllocatedStorage>\n                        <DBInstanceClass>db.m1.large</DBInstanceClass>\n                        <MasterUsername>master</MasterUsername>\n                </DBInstance>\n            </CreateDBInstanceResult>\n            <ResponseMetadata>\n                <RequestId>2e5d4270-8501-11e0-bd9b-a7b1ece36d51</RequestId>\n            </ResponseMetadata>\n        </CreateDBInstanceResponse>\n        '

    def test_create_db_instance_param_group_name(self):
        self.set_http_response(status_code=200)
        db = self.service_connection.create_dbinstance('SimCoProd01', 10, 'db.m1.large', 'master', 'Password01', param_group='default.mysql5.1', db_subnet_group_name='dbSubnetgroup01', backup_retention_period=0)
        self.assert_request_parameters({'Action': 'CreateDBInstance', 'AllocatedStorage': 10, 'AutoMinorVersionUpgrade': 'true', 'BackupRetentionPeriod': 0, 'DBInstanceClass': 'db.m1.large', 'DBInstanceIdentifier': 'SimCoProd01', 'DBParameterGroupName': 'default.mysql5.1', 'DBSubnetGroupName': 'dbSubnetgroup01', 'Engine': 'MySQL5.1', 'MasterUsername': 'master', 'MasterUserPassword': 'Password01', 'Port': 3306}, ignore_params_values=['Version'])
        self.assertEqual(db.id, 'simcoprod01')
        self.assertEqual(db.engine, 'mysql')
        self.assertEqual(db.status, 'creating')
        self.assertEqual(db.allocated_storage, 10)
        self.assertEqual(db.instance_class, 'db.m1.large')
        self.assertEqual(db.master_username, 'master')
        self.assertEqual(db.multi_az, False)
        self.assertEqual(db.pending_modified_values, {'MasterUserPassword': '****'})
        self.assertEqual(db.parameter_group.name, 'default.mysql5.1')
        self.assertEqual(db.parameter_group.description, None)
        self.assertEqual(db.parameter_group.engine, None)
        self.assertEqual(db.backup_retention_period, 0)

    def test_create_db_instance_param_group_instance(self):
        self.set_http_response(status_code=200)
        param_group = ParameterGroup()
        param_group.name = 'default.mysql5.1'
        db = self.service_connection.create_dbinstance('SimCoProd01', 10, 'db.m1.large', 'master', 'Password01', param_group=param_group, db_subnet_group_name='dbSubnetgroup01')
        self.assert_request_parameters({'Action': 'CreateDBInstance', 'AllocatedStorage': 10, 'AutoMinorVersionUpgrade': 'true', 'DBInstanceClass': 'db.m1.large', 'DBInstanceIdentifier': 'SimCoProd01', 'DBParameterGroupName': 'default.mysql5.1', 'DBSubnetGroupName': 'dbSubnetgroup01', 'Engine': 'MySQL5.1', 'MasterUsername': 'master', 'MasterUserPassword': 'Password01', 'Port': 3306}, ignore_params_values=['Version'])
        self.assertEqual(db.id, 'simcoprod01')
        self.assertEqual(db.engine, 'mysql')
        self.assertEqual(db.status, 'creating')
        self.assertEqual(db.allocated_storage, 10)
        self.assertEqual(db.instance_class, 'db.m1.large')
        self.assertEqual(db.master_username, 'master')
        self.assertEqual(db.multi_az, False)
        self.assertEqual(db.pending_modified_values, {'MasterUserPassword': '****'})
        self.assertEqual(db.parameter_group.name, 'default.mysql5.1')
        self.assertEqual(db.parameter_group.description, None)
        self.assertEqual(db.parameter_group.engine, None)