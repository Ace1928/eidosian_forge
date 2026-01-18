class InactiveDueCreditCollector(object):
    """Just a stub at the Collector which would not do anything"""

    def _donothing(self, *args, **kwargs):
        """Perform no good and no bad"""
        pass

    def dcite(self, *args, **kwargs):
        """If I could cite I would"""

        def nondecorating_decorator(func):
            return func
        return nondecorating_decorator
    cite = load = add = _donothing

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)