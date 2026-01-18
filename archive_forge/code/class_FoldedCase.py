import re
import itertools
import textwrap
import functools
from jaraco.functools import compose, method_cache
from jaraco.context import ExceptionTrap
class FoldedCase(str):
    """
    A case insensitive string class; behaves just like str
    except compares equal when the only variation is case.

    >>> s = FoldedCase('hello world')

    >>> s == 'Hello World'
    True

    >>> 'Hello World' == s
    True

    >>> s != 'Hello World'
    False

    >>> s.index('O')
    4

    >>> s.split('O')
    ['hell', ' w', 'rld']

    >>> sorted(map(FoldedCase, ['GAMMA', 'alpha', 'Beta']))
    ['alpha', 'Beta', 'GAMMA']

    Sequence membership is straightforward.

    >>> "Hello World" in [s]
    True
    >>> s in ["Hello World"]
    True

    Allows testing for set inclusion, but candidate and elements
    must both be folded.

    >>> FoldedCase("Hello World") in {s}
    True
    >>> s in {FoldedCase("Hello World")}
    True

    String inclusion works as long as the FoldedCase object
    is on the right.

    >>> "hello" in FoldedCase("Hello World")
    True

    But not if the FoldedCase object is on the left:

    >>> FoldedCase('hello') in 'Hello World'
    False

    In that case, use ``in_``:

    >>> FoldedCase('hello').in_('Hello World')
    True

    >>> FoldedCase('hello') > FoldedCase('Hello')
    False

    >>> FoldedCase('ß') == FoldedCase('ss')
    True
    """

    def __lt__(self, other):
        return self.casefold() < other.casefold()

    def __gt__(self, other):
        return self.casefold() > other.casefold()

    def __eq__(self, other):
        return self.casefold() == other.casefold()

    def __ne__(self, other):
        return self.casefold() != other.casefold()

    def __hash__(self):
        return hash(self.casefold())

    def __contains__(self, other):
        return super().casefold().__contains__(other.casefold())

    def in_(self, other):
        """Does self appear in other?"""
        return self in FoldedCase(other)

    @method_cache
    def casefold(self):
        return super().casefold()

    def index(self, sub):
        return self.casefold().index(sub.casefold())

    def split(self, splitter=' ', maxsplit=0):
        pattern = re.compile(re.escape(splitter), re.I)
        return pattern.split(self, maxsplit)