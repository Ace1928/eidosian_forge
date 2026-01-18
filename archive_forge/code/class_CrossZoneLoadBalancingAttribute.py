class CrossZoneLoadBalancingAttribute(object):
    """
    Represents the CrossZoneLoadBalancing segement of ELB Attributes.
    """

    def __init__(self, connection=None):
        self.enabled = None

    def __repr__(self):
        return 'CrossZoneLoadBalancingAttribute(%s)' % self.enabled

    def startElement(self, name, attrs, connection):
        pass

    def endElement(self, name, value, connection):
        if name == 'Enabled':
            if value.lower() == 'true':
                self.enabled = True
            else:
                self.enabled = False