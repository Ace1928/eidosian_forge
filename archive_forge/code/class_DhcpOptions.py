from boto.ec2.ec2object import TaggedEC2Object
class DhcpOptions(TaggedEC2Object):

    def __init__(self, connection=None):
        super(DhcpOptions, self).__init__(connection)
        self.id = None
        self.options = None

    def __repr__(self):
        return 'DhcpOptions:%s' % self.id

    def startElement(self, name, attrs, connection):
        retval = super(DhcpOptions, self).startElement(name, attrs, connection)
        if retval is not None:
            return retval
        if name == 'dhcpConfigurationSet':
            self.options = DhcpConfigSet()
            return self.options

    def endElement(self, name, value, connection):
        if name == 'dhcpOptionsId':
            self.id = value
        else:
            setattr(self, name, value)