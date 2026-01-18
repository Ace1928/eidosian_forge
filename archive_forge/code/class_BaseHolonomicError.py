class BaseHolonomicError(Exception):

    def new(self, *args):
        raise NotImplementedError('abstract base class')