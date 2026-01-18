from tests.unit import unittest
from tests.unit import AWSMockServiceTestCase
from boto.ec2.securitygroup import SecurityGroup
from boto.rds import RDSConnection
from boto.rds.vpcsecuritygroupmembership import VPCSecurityGroupMembership
from boto.rds.parametergroup import ParameterGroup
from boto.rds.logfile import LogFile, LogFileObject
import xml.sax.saxutils as saxutils
class TestRDSConnection(AWSMockServiceTestCase):
    connection_class = RDSConnection

    def setUp(self):
        super(TestRDSConnection, self).setUp()

    def default_body(self):
        return '\n        <DescribeDBInstancesResponse>\n          <DescribeDBInstancesResult>\n            <DBInstances>\n                <DBInstance>\n                  <Iops>2000</Iops>\n                  <BackupRetentionPeriod>1</BackupRetentionPeriod>\n                  <MultiAZ>false</MultiAZ>\n                  <DBInstanceStatus>backing-up</DBInstanceStatus>\n                  <DBInstanceIdentifier>mydbinstance2</DBInstanceIdentifier>\n                  <PreferredBackupWindow>10:30-11:00</PreferredBackupWindow>\n                  <PreferredMaintenanceWindow>wed:06:30-wed:07:00</PreferredMaintenanceWindow>\n                  <OptionGroupMembership>\n                    <OptionGroupName>default:mysql-5-5</OptionGroupName>\n                    <Status>in-sync</Status>\n                  </OptionGroupMembership>\n                  <AvailabilityZone>us-west-2b</AvailabilityZone>\n                  <ReadReplicaDBInstanceIdentifiers/>\n                  <Engine>mysql</Engine>\n                  <PendingModifiedValues/>\n                  <LicenseModel>general-public-license</LicenseModel>\n                  <DBParameterGroups>\n                    <DBParameterGroup>\n                      <ParameterApplyStatus>in-sync</ParameterApplyStatus>\n                      <DBParameterGroupName>default.mysql5.5</DBParameterGroupName>\n                    </DBParameterGroup>\n                  </DBParameterGroups>\n                  <Endpoint>\n                    <Port>3306</Port>\n                    <Address>mydbinstance2.c0hjqouvn9mf.us-west-2.rds.amazonaws.com</Address>\n                  </Endpoint>\n                  <EngineVersion>5.5.27</EngineVersion>\n                  <DBSecurityGroups>\n                    <DBSecurityGroup>\n                      <Status>active</Status>\n                      <DBSecurityGroupName>default</DBSecurityGroupName>\n                    </DBSecurityGroup>\n                  </DBSecurityGroups>\n                  <VpcSecurityGroups>\n                    <VpcSecurityGroupMembership>\n                      <VpcSecurityGroupId>sg-1</VpcSecurityGroupId>\n                      <Status>active</Status>\n                    </VpcSecurityGroupMembership>\n                  </VpcSecurityGroups>\n                  <DBName>mydb2</DBName>\n                  <AutoMinorVersionUpgrade>true</AutoMinorVersionUpgrade>\n                  <InstanceCreateTime>2012-10-03T22:01:51.047Z</InstanceCreateTime>\n                  <AllocatedStorage>200</AllocatedStorage>\n                  <DBInstanceClass>db.m1.large</DBInstanceClass>\n                  <MasterUsername>awsuser</MasterUsername>\n                  <StatusInfos>\n                    <DBInstanceStatusInfo>\n                      <Message></Message>\n                      <Normal>true</Normal>\n                      <Status>replicating</Status>\n                      <StatusType>read replication</StatusType>\n                    </DBInstanceStatusInfo>\n                  </StatusInfos>\n                  <DBSubnetGroup>\n                    <VpcId>990524496922</VpcId>\n                    <SubnetGroupStatus>Complete</SubnetGroupStatus>\n                    <DBSubnetGroupDescription>My modified DBSubnetGroup</DBSubnetGroupDescription>\n                    <DBSubnetGroupName>mydbsubnetgroup</DBSubnetGroupName>\n                    <Subnets>\n                      <Subnet>\n                        <SubnetStatus>Active</SubnetStatus>\n                        <SubnetIdentifier>subnet-7c5b4115</SubnetIdentifier>\n                        <SubnetAvailabilityZone>\n                        <Name>us-east-1c</Name>\n                      </SubnetAvailabilityZone>\n                      </Subnet>\n                      <Subnet>\n                        <SubnetStatus>Active</SubnetStatus>\n                        <SubnetIdentifier>subnet-7b5b4112</SubnetIdentifier>\n                        <SubnetAvailabilityZone>\n                          <Name>us-east-1b</Name>\n                        </SubnetAvailabilityZone>\n                      </Subnet>\n                      <Subnet>\n                        <SubnetStatus>Active</SubnetStatus>\n                        <SubnetIdentifier>subnet-3ea6bd57</SubnetIdentifier>\n                        <SubnetAvailabilityZone>\n                          <Name>us-east-1d</Name>\n                        </SubnetAvailabilityZone>\n                      </Subnet>\n                    </Subnets>\n                  </DBSubnetGroup>  \n              </DBInstance>\n            </DBInstances>\n          </DescribeDBInstancesResult>\n        </DescribeDBInstancesResponse>\n        '

    def test_get_all_db_instances(self):
        self.set_http_response(status_code=200)
        response = self.service_connection.get_all_dbinstances('instance_id')
        self.assertEqual(len(response), 1)
        self.assert_request_parameters({'Action': 'DescribeDBInstances', 'DBInstanceIdentifier': 'instance_id'}, ignore_params_values=['Version'])
        db = response[0]
        self.assertEqual(db.id, 'mydbinstance2')
        self.assertEqual(db.create_time, '2012-10-03T22:01:51.047Z')
        self.assertEqual(db.engine, 'mysql')
        self.assertEqual(db.status, 'backing-up')
        self.assertEqual(db.allocated_storage, 200)
        self.assertEqual(db.endpoint, (u'mydbinstance2.c0hjqouvn9mf.us-west-2.rds.amazonaws.com', 3306))
        self.assertEqual(db.instance_class, 'db.m1.large')
        self.assertEqual(db.master_username, 'awsuser')
        self.assertEqual(db.availability_zone, 'us-west-2b')
        self.assertEqual(db.backup_retention_period, 1)
        self.assertEqual(db.preferred_backup_window, '10:30-11:00')
        self.assertEqual(db.preferred_maintenance_window, 'wed:06:30-wed:07:00')
        self.assertEqual(db.latest_restorable_time, None)
        self.assertEqual(db.multi_az, False)
        self.assertEqual(db.iops, 2000)
        self.assertEqual(db.pending_modified_values, {})
        self.assertEqual(db.parameter_group.name, 'default.mysql5.5')
        self.assertEqual(db.parameter_group.description, None)
        self.assertEqual(db.parameter_group.engine, None)
        self.assertEqual(db.security_group.owner_id, None)
        self.assertEqual(db.security_group.name, 'default')
        self.assertEqual(db.security_group.description, None)
        self.assertEqual(db.security_group.ec2_groups, [])
        self.assertEqual(db.security_group.ip_ranges, [])
        self.assertEqual(len(db.status_infos), 1)
        self.assertEqual(db.status_infos[0].message, '')
        self.assertEqual(db.status_infos[0].normal, True)
        self.assertEqual(db.status_infos[0].status, 'replicating')
        self.assertEqual(db.status_infos[0].status_type, 'read replication')
        self.assertEqual(db.vpc_security_groups[0].status, 'active')
        self.assertEqual(db.vpc_security_groups[0].vpc_group, 'sg-1')
        self.assertEqual(db.license_model, 'general-public-license')
        self.assertEqual(db.engine_version, '5.5.27')
        self.assertEqual(db.auto_minor_version_upgrade, True)
        self.assertEqual(db.subnet_group.name, 'mydbsubnetgroup')