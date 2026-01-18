from boto.regioninfo import RegionInfo
class SQSRegionInfo(RegionInfo):

    def __init__(self, connection=None, name=None, endpoint=None, connection_cls=None):
        from boto.sqs.connection import SQSConnection
        super(SQSRegionInfo, self).__init__(connection, name, endpoint, SQSConnection)