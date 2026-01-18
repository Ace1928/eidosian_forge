from boto.rds.dbsecuritygroup import DBSecurityGroup
from boto.rds.parametergroup import ParameterGroup
from boto.rds.statusinfo import StatusInfo
from boto.rds.dbsubnetgroup import DBSubnetGroup
from boto.rds.vpcsecuritygroupmembership import VPCSecurityGroupMembership
from boto.resultset import ResultSet
class ReadReplicaDBInstanceIdentifiers(list):

    def startElement(self, name, attrs, connection):
        return None

    def endElement(self, name, value, connection):
        if name == 'ReadReplicaDBInstanceIdentifier':
            self.append(value)