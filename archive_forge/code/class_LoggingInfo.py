class LoggingInfo(object):

    def __init__(self, bucket='', prefix=''):
        self.bucket = bucket
        self.prefix = prefix

    def startElement(self, name, attrs, connection):
        return None

    def endElement(self, name, value, connection):
        if name == 'Bucket':
            self.bucket = value
        elif name == 'Prefix':
            self.prefix = value
        else:
            setattr(self, name, value)