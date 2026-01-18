from boto.resultset import ResultSet
class BootstrapActionList(EmrObject):
    Fields = set(['Marker'])

    def __init__(self, connection=None):
        self.connection = connection
        self.actions = None

    def startElement(self, name, attrs, connection):
        if name == 'BootstrapActions':
            self.actions = ResultSet([('member', BootstrapAction)])
            return self.actions
        else:
            return None