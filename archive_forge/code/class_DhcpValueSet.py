from boto.ec2.ec2object import TaggedEC2Object
class DhcpValueSet(list):

    def startElement(self, name, attrs, connection):
        pass

    def endElement(self, name, value, connection):
        if name == 'value':
            self.append(value)