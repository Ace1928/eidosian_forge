import warnings
from io import StringIO
class IncludedResponse(object):

    def __init__(self):
        self.headers = None
        self.status = None
        self.output = StringIO()
        self.str = None

    def close(self):
        self.str = self.output.getvalue()
        self.output.close()
        self.output = None

    def write(self, s):
        assert self.output is not None, 'This response has already been closed and no further data can be written.'
        self.output.write(s)

    def __str__(self):
        return self.body

    def body__get(self):
        if self.str is None:
            return self.output.getvalue()
        else:
            return self.str
    body = property(body__get)