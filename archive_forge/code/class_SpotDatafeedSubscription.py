from boto.ec2.ec2object import EC2Object
from boto.ec2.spotinstancerequest import SpotInstanceStateFault
class SpotDatafeedSubscription(EC2Object):

    def __init__(self, connection=None, owner_id=None, bucket=None, prefix=None, state=None, fault=None):
        super(SpotDatafeedSubscription, self).__init__(connection)
        self.owner_id = owner_id
        self.bucket = bucket
        self.prefix = prefix
        self.state = state
        self.fault = fault

    def __repr__(self):
        return 'SpotDatafeedSubscription:%s' % self.bucket

    def startElement(self, name, attrs, connection):
        if name == 'fault':
            self.fault = SpotInstanceStateFault()
            return self.fault
        else:
            return None

    def endElement(self, name, value, connection):
        if name == 'ownerId':
            self.owner_id = value
        elif name == 'bucket':
            self.bucket = value
        elif name == 'prefix':
            self.prefix = value
        elif name == 'state':
            self.state = value
        else:
            setattr(self, name, value)

    def delete(self, dry_run=False):
        return self.connection.delete_spot_datafeed_subscription(dry_run=dry_run)