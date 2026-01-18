import types
from ._impl import (
class MatchesPredicate(Matcher):
    """Match if a given function returns True.

    It is reasonably common to want to make a very simple matcher based on a
    function that you already have that returns True or False given a single
    argument (i.e. a predicate function).  This matcher makes it very easy to
    do so. e.g.::

      IsEven = MatchesPredicate(lambda x: x % 2 == 0, '%s is not even')
      self.assertThat(4, IsEven)
    """

    def __init__(self, predicate, message):
        """Create a ``MatchesPredicate`` matcher.

        :param predicate: A function that takes a single argument and returns
            a value that will be interpreted as a boolean.
        :param message: A message to describe a mismatch.  It will be formatted
            with '%' and be given whatever was passed to ``match()``. Thus, it
            needs to contain exactly one thing like '%s', '%d' or '%f'.
        """
        self.predicate = predicate
        self.message = message

    def __str__(self):
        return '{}({!r}, {!r})'.format(self.__class__.__name__, self.predicate, self.message)

    def match(self, x):
        if not self.predicate(x):
            return Mismatch(self.message % x)