import io
import os
class IterUnreader(Unreader):

    def __init__(self, iterable):
        super().__init__()
        self.iter = iter(iterable)

    def chunk(self):
        if not self.iter:
            return b''
        try:
            return next(self.iter)
        except StopIteration:
            self.iter = None
            return b''