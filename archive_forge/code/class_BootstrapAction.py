from boto.resultset import ResultSet
class BootstrapAction(EmrObject):
    Fields = set(['Args', 'Name', 'Path', 'ScriptPath'])

    def startElement(self, name, attrs, connection):
        if name == 'Args':
            self.args = ResultSet([('member', Arg)])
            return self.args