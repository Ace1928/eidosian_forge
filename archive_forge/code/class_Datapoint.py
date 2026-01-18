from datetime import datetime
class Datapoint(dict):

    def __init__(self, connection=None):
        dict.__init__(self)
        self.connection = connection

    def startElement(self, name, attrs, connection):
        pass

    def endElement(self, name, value, connection):
        if name in ['Average', 'Maximum', 'Minimum', 'Sum', 'SampleCount']:
            self[name] = float(value)
        elif name == 'Timestamp':
            self[name] = datetime.strptime(value, '%Y-%m-%dT%H:%M:%SZ')
        elif name != 'member':
            self[name] = value