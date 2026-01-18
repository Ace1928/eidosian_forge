from boto.regioninfo import RegionInfo
class RDSRegionInfo(RegionInfo):

    def __init__(self, connection=None, name=None, endpoint=None, connection_cls=None):
        from boto.rds import RDSConnection
        super(RDSRegionInfo, self).__init__(connection, name, endpoint, RDSConnection)