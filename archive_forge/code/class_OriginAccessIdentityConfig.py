import uuid
class OriginAccessIdentityConfig(object):

    def __init__(self, connection=None, caller_reference='', comment=''):
        self.connection = connection
        if caller_reference:
            self.caller_reference = caller_reference
        else:
            self.caller_reference = str(uuid.uuid4())
        self.comment = comment

    def to_xml(self):
        s = '<?xml version="1.0" encoding="UTF-8"?>\n'
        s += '<CloudFrontOriginAccessIdentityConfig xmlns="http://cloudfront.amazonaws.com/doc/2009-09-09/">\n'
        s += '  <CallerReference>%s</CallerReference>\n' % self.caller_reference
        if self.comment:
            s += '  <Comment>%s</Comment>\n' % self.comment
        s += '</CloudFrontOriginAccessIdentityConfig>\n'
        return s

    def startElement(self, name, attrs, connection):
        return None

    def endElement(self, name, value, connection):
        if name == 'Comment':
            self.comment = value
        elif name == 'CallerReference':
            self.caller_reference = value
        else:
            setattr(self, name, value)