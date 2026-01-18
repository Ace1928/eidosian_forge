class ConnectionDrainingAttribute(object):
    """
    Represents the ConnectionDraining segment of ELB attributes.
    """

    def __init__(self, connection=None):
        self.enabled = None
        self.timeout = None

    def __repr__(self):
        return 'ConnectionDraining(%s, %s)' % (self.enabled, self.timeout)

    def startElement(self, name, attrs, connection):
        pass

    def endElement(self, name, value, connection):
        if name == 'Enabled':
            if value.lower() == 'true':
                self.enabled = True
            else:
                self.enabled = False
        elif name == 'Timeout':
            self.timeout = int(value)