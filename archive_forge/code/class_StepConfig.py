from boto.resultset import ResultSet
class StepConfig(EmrObject):
    Fields = set(['Jar', 'MainClass'])

    def __init__(self, connection=None):
        self.connection = connection
        self.properties = None
        self.args = None

    def startElement(self, name, attrs, connection):
        if name == 'Properties':
            self.properties = ResultSet([('member', KeyValue)])
            return self.properties
        elif name == 'Args':
            self.args = ResultSet([('member', Arg)])
            return self.args
        else:
            return None