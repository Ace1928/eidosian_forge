class InvalidChunkSize(IOError):

    def __init__(self, data):
        self.data = data

    def __str__(self):
        return 'Invalid chunk size: %r' % self.data