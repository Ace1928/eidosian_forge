class ChunkMissingTerminator(IOError):

    def __init__(self, term):
        self.term = term

    def __str__(self):
        return "Invalid chunk terminator is not '\\r\\n': %r" % self.term