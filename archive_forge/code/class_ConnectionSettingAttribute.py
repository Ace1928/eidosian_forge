class ConnectionSettingAttribute(object):
    """
    Represents the ConnectionSetting segment of ELB Attributes.
    """

    def __init__(self, connection=None):
        self.idle_timeout = None

    def __repr__(self):
        return 'ConnectionSettingAttribute(%s)' % self.idle_timeout

    def startElement(self, name, attrs, connection):
        pass

    def endElement(self, name, value, connection):
        if name == 'IdleTimeout':
            self.idle_timeout = int(value)