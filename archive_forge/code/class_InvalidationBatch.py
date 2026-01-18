import uuid
from boto.compat import urllib
from boto.resultset import ResultSet
class InvalidationBatch(object):
    """A simple invalidation request.
        :see: http://docs.amazonwebservices.com/AmazonCloudFront/2010-08-01/APIReference/index.html?InvalidationBatchDatatype.html
    """

    def __init__(self, paths=None, connection=None, distribution=None, caller_reference=''):
        """Create a new invalidation request:
            :paths: An array of paths to invalidate
        """
        self.paths = paths or []
        self.distribution = distribution
        self.caller_reference = caller_reference
        if not self.caller_reference:
            self.caller_reference = str(uuid.uuid4())
        if distribution:
            self.connection = distribution
        else:
            self.connection = connection

    def __repr__(self):
        return '<InvalidationBatch: %s>' % self.id

    def add(self, path):
        """Add another path to this invalidation request"""
        return self.paths.append(path)

    def remove(self, path):
        """Remove a path from this invalidation request"""
        return self.paths.remove(path)

    def __iter__(self):
        return iter(self.paths)

    def __getitem__(self, i):
        return self.paths[i]

    def __setitem__(self, k, v):
        self.paths[k] = v

    def escape(self, p):
        """Escape a path, make sure it begins with a slash and contains no invalid characters. Retain literal wildcard characters."""
        if not p[0] == '/':
            p = '/%s' % p
        return urllib.parse.quote(p, safe='/*')

    def to_xml(self):
        """Get this batch as XML"""
        assert self.connection is not None
        s = '<?xml version="1.0" encoding="UTF-8"?>\n'
        s += '<InvalidationBatch xmlns="http://cloudfront.amazonaws.com/doc/%s/">\n' % self.connection.Version
        for p in self.paths:
            s += '    <Path>%s</Path>\n' % self.escape(p)
        s += '    <CallerReference>%s</CallerReference>\n' % self.caller_reference
        s += '</InvalidationBatch>\n'
        return s

    def startElement(self, name, attrs, connection):
        if name == 'InvalidationBatch':
            self.paths = []
        return None

    def endElement(self, name, value, connection):
        if name == 'Path':
            self.paths.append(value)
        elif name == 'Status':
            self.status = value
        elif name == 'Id':
            self.id = value
        elif name == 'CreateTime':
            self.create_time = value
        elif name == 'CallerReference':
            self.caller_reference = value
        return None