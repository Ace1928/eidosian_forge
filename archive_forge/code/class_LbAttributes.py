class LbAttributes(object):
    """
    Represents the Attributes of an Elastic Load Balancer.
    """

    def __init__(self, connection=None):
        self.connection = connection
        self.cross_zone_load_balancing = CrossZoneLoadBalancingAttribute(self.connection)
        self.access_log = AccessLogAttribute(self.connection)
        self.connection_draining = ConnectionDrainingAttribute(self.connection)
        self.connecting_settings = ConnectionSettingAttribute(self.connection)

    def __repr__(self):
        return 'LbAttributes(%s, %s, %s, %s)' % (repr(self.cross_zone_load_balancing), repr(self.access_log), repr(self.connection_draining), repr(self.connecting_settings))

    def startElement(self, name, attrs, connection):
        if name == 'CrossZoneLoadBalancing':
            return self.cross_zone_load_balancing
        if name == 'AccessLog':
            return self.access_log
        if name == 'ConnectionDraining':
            return self.connection_draining
        if name == 'ConnectionSettings':
            return self.connecting_settings

    def endElement(self, name, value, connection):
        pass