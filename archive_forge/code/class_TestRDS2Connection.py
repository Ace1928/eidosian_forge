from tests.unit import unittest
from tests.unit import AWSMockServiceTestCase
from boto.ec2.securitygroup import SecurityGroup
from boto.rds2.layer1 import RDSConnection
class TestRDS2Connection(AWSMockServiceTestCase):
    connection_class = RDSConnection

    def setUp(self):
        super(TestRDS2Connection, self).setUp()

    def default_body(self):
        return '{\n    "DescribeDBInstancesResponse": {\n        "DescribeDBInstancesResult": {\n            "DBInstances": [{\n                "DBInstance": {\n                    "Iops": 2000,\n                    "BackupRetentionPeriod": 1,\n                    "MultiAZ": false,\n                    "DBInstanceStatus": "backing-up",\n                    "DBInstanceIdentifier": "mydbinstance2",\n                    "PreferredBackupWindow": "10:30-11:00",\n                    "PreferredMaintenanceWindow": "wed:06:30-wed:07:00",\n                    "OptionGroupMembership": {\n                        "OptionGroupName": "default:mysql-5-5",\n                        "Status": "in-sync"\n                    },\n                    "AvailabilityZone": "us-west-2b",\n                    "ReadReplicaDBInstanceIdentifiers": null,\n                    "Engine": "mysql",\n                    "PendingModifiedValues": null,\n                    "LicenseModel": "general-public-license",\n                    "DBParameterGroups": [{\n                        "DBParameterGroup": {\n                            "ParameterApplyStatus": "in-sync",\n                            "DBParameterGroupName": "default.mysql5.5"\n                        }\n                    }],\n                    "Endpoint": {\n                        "Port": 3306,\n                        "Address": "mydbinstance2.c0hjqouvn9mf.us-west-2.rds.amazonaws.com"\n                    },\n                    "EngineVersion": "5.5.27",\n                    "DBSecurityGroups": [{\n                        "DBSecurityGroup": {\n                            "Status": "active",\n                            "DBSecurityGroupName": "default"\n                        }\n                    }],\n                    "VpcSecurityGroups": [{\n                        "VpcSecurityGroupMembership": {\n                            "VpcSecurityGroupId": "sg-1",\n                            "Status": "active"\n                        }\n                    }],\n                    "DBName": "mydb2",\n                    "AutoMinorVersionUpgrade": true,\n                    "InstanceCreateTime": "2012-10-03T22:01:51.047Z",\n                    "AllocatedStorage": 200,\n                    "DBInstanceClass": "db.m1.large",\n                    "MasterUsername": "awsuser",\n                    "StatusInfos": [{\n                        "DBInstanceStatusInfo": {\n                            "Message": null,\n                            "Normal": true,\n                            "Status": "replicating",\n                            "StatusType": "read replication"\n                        }\n                    }],\n                    "DBSubnetGroup": {\n                        "VpcId": "990524496922",\n                        "SubnetGroupStatus": "Complete",\n                        "DBSubnetGroupDescription": "My modified DBSubnetGroup",\n                        "DBSubnetGroupName": "mydbsubnetgroup",\n                        "Subnets": [{\n                            "Subnet": {\n                                "SubnetStatus": "Active",\n                                "SubnetIdentifier": "subnet-7c5b4115",\n                                "SubnetAvailabilityZone": {\n                                    "Name": "us-east-1c"\n                                }\n                            },\n                            "Subnet": {\n                                "SubnetStatus": "Active",\n                                "SubnetIdentifier": "subnet-7b5b4112",\n                                "SubnetAvailabilityZone": {\n                                    "Name": "us-east-1b"\n                                }\n                            },\n                            "Subnet": {\n                                "SubnetStatus": "Active",\n                                "SubnetIdentifier": "subnet-3ea6bd57",\n                                "SubnetAvailabilityZone": {\n                                    "Name": "us-east-1d"\n                                }\n                            }\n                        }]\n                    }\n                }\n            }]\n        }\n    }\n    }'

    def test_describe_db_instances(self):
        self.set_http_response(status_code=200)
        response = self.service_connection.describe_db_instances('instance_id')
        self.assertEqual(len(response), 1)
        self.assert_request_parameters({'Action': 'DescribeDBInstances', 'ContentType': 'JSON', 'DBInstanceIdentifier': 'instance_id'}, ignore_params_values=['Version'])
        db = response['DescribeDBInstancesResponse']['DescribeDBInstancesResult']['DBInstances'][0]['DBInstance']
        self.assertEqual(db['DBInstanceIdentifier'], 'mydbinstance2')
        self.assertEqual(db['InstanceCreateTime'], '2012-10-03T22:01:51.047Z')
        self.assertEqual(db['Engine'], 'mysql')
        self.assertEqual(db['DBInstanceStatus'], 'backing-up')
        self.assertEqual(db['AllocatedStorage'], 200)
        self.assertEqual(db['Endpoint']['Port'], 3306)
        self.assertEqual(db['DBInstanceClass'], 'db.m1.large')
        self.assertEqual(db['MasterUsername'], 'awsuser')
        self.assertEqual(db['AvailabilityZone'], 'us-west-2b')
        self.assertEqual(db['BackupRetentionPeriod'], 1)
        self.assertEqual(db['PreferredBackupWindow'], '10:30-11:00')
        self.assertEqual(db['PreferredMaintenanceWindow'], 'wed:06:30-wed:07:00')
        self.assertEqual(db['MultiAZ'], False)
        self.assertEqual(db['Iops'], 2000)
        self.assertEqual(db['PendingModifiedValues'], None)
        self.assertEqual(db['DBParameterGroups'][0]['DBParameterGroup']['DBParameterGroupName'], 'default.mysql5.5')
        self.assertEqual(db['DBSecurityGroups'][0]['DBSecurityGroup']['DBSecurityGroupName'], 'default')
        self.assertEqual(db['DBSecurityGroups'][0]['DBSecurityGroup']['Status'], 'active')
        self.assertEqual(len(db['StatusInfos']), 1)
        self.assertEqual(db['StatusInfos'][0]['DBInstanceStatusInfo']['Message'], None)
        self.assertEqual(db['StatusInfos'][0]['DBInstanceStatusInfo']['Normal'], True)
        self.assertEqual(db['StatusInfos'][0]['DBInstanceStatusInfo']['Status'], 'replicating')
        self.assertEqual(db['StatusInfos'][0]['DBInstanceStatusInfo']['StatusType'], 'read replication')
        self.assertEqual(db['VpcSecurityGroups'][0]['VpcSecurityGroupMembership']['Status'], 'active')
        self.assertEqual(db['VpcSecurityGroups'][0]['VpcSecurityGroupMembership']['VpcSecurityGroupId'], 'sg-1')
        self.assertEqual(db['LicenseModel'], 'general-public-license')
        self.assertEqual(db['EngineVersion'], '5.5.27')
        self.assertEqual(db['AutoMinorVersionUpgrade'], True)
        self.assertEqual(db['DBSubnetGroup']['DBSubnetGroupName'], 'mydbsubnetgroup')