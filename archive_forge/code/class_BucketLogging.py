import xml.sax.saxutils
from boto.s3.acl import Grant
class BucketLogging(object):

    def __init__(self, target=None, prefix=None, grants=None):
        self.target = target
        self.prefix = prefix
        if grants is None:
            self.grants = []
        else:
            self.grants = grants

    def __repr__(self):
        if self.target is None:
            return '<BucketLoggingStatus: Disabled>'
        grants = []
        for g in self.grants:
            if g.type == 'CanonicalUser':
                u = g.display_name
            elif g.type == 'Group':
                u = g.uri
            else:
                u = g.email_address
            grants.append('%s = %s' % (u, g.permission))
        return '<BucketLoggingStatus: %s/%s (%s)>' % (self.target, self.prefix, ', '.join(grants))

    def add_grant(self, grant):
        self.grants.append(grant)

    def startElement(self, name, attrs, connection):
        if name == 'Grant':
            self.grants.append(Grant())
            return self.grants[-1]
        else:
            return None

    def endElement(self, name, value, connection):
        if name == 'TargetBucket':
            self.target = value
        elif name == 'TargetPrefix':
            self.prefix = value
        else:
            setattr(self, name, value)

    def to_xml(self):
        s = u'<?xml version="1.0" encoding="UTF-8"?>'
        s += u'<BucketLoggingStatus xmlns="http://doc.s3.amazonaws.com/2006-03-01">'
        if self.target is not None:
            s += u'<LoggingEnabled>'
            s += u'<TargetBucket>%s</TargetBucket>' % self.target
            prefix = self.prefix or ''
            s += u'<TargetPrefix>%s</TargetPrefix>' % xml.sax.saxutils.escape(prefix)
            if self.grants:
                s += '<TargetGrants>'
                for grant in self.grants:
                    s += grant.to_xml()
                s += '</TargetGrants>'
            s += u'</LoggingEnabled>'
        s += u'</BucketLoggingStatus>'
        return s