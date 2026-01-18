from boto.regioninfo import RegionInfo
class EC2RegionInfo(RegionInfo):
    """
    Represents an EC2 Region
    """

    def __init__(self, connection=None, name=None, endpoint=None, connection_cls=None):
        from boto.ec2.connection import EC2Connection
        super(EC2RegionInfo, self).__init__(connection, name, endpoint, EC2Connection)