class NoMoreData(IOError):

    def __init__(self, buf=None):
        self.buf = buf

    def __str__(self):
        return 'No more data after: %r' % self.buf