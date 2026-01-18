from boto.resultset import ResultSet
from boto.ec2.ec2object import EC2Object
from boto.utils import parse_ts
class ReservedInstance(ReservedInstancesOffering):

    def __init__(self, connection=None, id=None, instance_type=None, availability_zone=None, duration=None, fixed_price=None, usage_price=None, description=None, instance_count=None, state=None):
        super(ReservedInstance, self).__init__(connection, id, instance_type, availability_zone, duration, fixed_price, usage_price, description)
        self.instance_count = instance_count
        self.state = state
        self.start = None
        self.end = None

    def __repr__(self):
        return 'ReservedInstance:%s' % self.id

    def endElement(self, name, value, connection):
        if name == 'reservedInstancesId':
            self.id = value
        if name == 'instanceCount':
            self.instance_count = int(value)
        elif name == 'state':
            self.state = value
        elif name == 'start':
            self.start = value
        elif name == 'end':
            self.end = value
        else:
            super(ReservedInstance, self).endElement(name, value, connection)