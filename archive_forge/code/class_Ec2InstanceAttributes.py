from boto.resultset import ResultSet
class Ec2InstanceAttributes(EmrObject):
    Fields = set(['Ec2KeyName', 'Ec2SubnetId', 'Ec2AvailabilityZone', 'IamInstanceProfile'])