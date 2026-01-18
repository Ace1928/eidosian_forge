from testtools.compat import (
class MismatchDecorator:
    """Decorate a ``Mismatch``.

    Forwards all messages to the original mismatch object.  Probably the best
    way to use this is inherit from this class and then provide your own
    custom decoration logic.
    """

    def __init__(self, original):
        """Construct a `MismatchDecorator`.

        :param original: A `Mismatch` object to decorate.
        """
        self.original = original

    def __repr__(self):
        return f'<testtools.matchers.MismatchDecorator({self.original!r})>'

    def describe(self):
        return self.original.describe()

    def get_details(self):
        return self.original.get_details()