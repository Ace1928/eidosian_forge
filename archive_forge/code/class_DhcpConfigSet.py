from boto.ec2.ec2object import TaggedEC2Object
class DhcpConfigSet(dict):

    def startElement(self, name, attrs, connection):
        if name == 'valueSet':
            if self._name not in self:
                self[self._name] = DhcpValueSet()
            return self[self._name]

    def endElement(self, name, value, connection):
        if name == 'key':
            self._name = value