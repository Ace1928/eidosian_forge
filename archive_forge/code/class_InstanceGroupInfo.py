from boto.resultset import ResultSet
class InstanceGroupInfo(EmrObject):
    Fields = set(['Id', 'Name', 'Market', 'InstanceGroupType', 'BidPrice', 'InstanceType', 'RequestedInstanceCount', 'RunningInstanceCount'])

    def __init__(self, connection=None):
        self.connection = connection
        self.status = None

    def startElement(self, name, attrs, connection):
        if name == 'Status':
            self.status = ClusterStatus()
            return self.status
        else:
            return None