from threading import Lock
class ResponsibleGenerator:
    """A generator that will help clean up when it is done being used."""
    __slots__ = ['cleanup', 'gen']

    def __init__(self, gen, cleanup):
        self.cleanup = cleanup
        self.gen = gen

    def __del__(self):
        self.cleanup()

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.gen)