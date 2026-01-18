from boto import handler
import xml.sax
class MultiDeleteResult(object):
    """
    The status returned from a MultiObject Delete request.

    :ivar deleted: A list of successfully deleted objects.  Note that if
        the quiet flag was specified in the request, this list will
        be empty because only error responses would be returned.

    :ivar errors: A list of unsuccessfully deleted objects.
    """

    def __init__(self, bucket=None):
        self.bucket = None
        self.deleted = []
        self.errors = []

    def startElement(self, name, attrs, connection):
        if name == 'Deleted':
            d = Deleted()
            self.deleted.append(d)
            return d
        elif name == 'Error':
            e = Error()
            self.errors.append(e)
            return e
        return None

    def endElement(self, name, value, connection):
        setattr(self, name, value)