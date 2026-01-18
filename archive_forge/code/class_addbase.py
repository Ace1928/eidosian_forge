import tempfile
class addbase(tempfile._TemporaryFileWrapper):
    """Base class for addinfo and addclosehook. Is a good idea for garbage collection."""

    def __init__(self, fp):
        super(addbase, self).__init__(fp, '<urllib response>', delete=False)
        self.fp = fp

    def __repr__(self):
        return '<%s at %r whose fp = %r>' % (self.__class__.__name__, id(self), self.file)

    def __enter__(self):
        if self.fp.closed:
            raise ValueError('I/O operation on closed file')
        return self

    def __exit__(self, type, value, traceback):
        self.close()