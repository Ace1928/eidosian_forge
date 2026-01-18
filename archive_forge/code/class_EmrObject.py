from boto.resultset import ResultSet
class EmrObject(object):
    Fields = set()

    def __init__(self, connection=None):
        self.connection = connection

    def startElement(self, name, attrs, connection):
        pass

    def endElement(self, name, value, connection):
        if name in self.Fields:
            setattr(self, name.lower(), value)