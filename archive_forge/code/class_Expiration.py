from boto.compat import six
class Expiration(object):
    """
    When an object will expire.

    :ivar days: The number of days until the object expires

    :ivar date: The date when the object will expire. Must be
        in ISO 8601 format.
    """

    def __init__(self, days=None, date=None):
        self.days = days
        self.date = date

    def startElement(self, name, attrs, connection):
        return None

    def endElement(self, name, value, connection):
        if name == 'Days':
            self.days = int(value)
        elif name == 'Date':
            self.date = value

    def __repr__(self):
        if self.days is None:
            how_long = 'on: %s' % self.date
        else:
            how_long = 'in: %s days' % self.days
        return '<Expiration: %s>' % how_long

    def to_xml(self):
        s = '<Expiration>'
        if self.days is not None:
            s += '<Days>%s</Days>' % self.days
        elif self.date is not None:
            s += '<Date>%s</Date>' % self.date
        s += '</Expiration>'
        return s