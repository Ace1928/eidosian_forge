from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
class Peeker(object):
    """Peeks the first element from an iterable.

  The returned object is another iterable that is equivalent to the original.
  If the object is not iterable then the first item is the object itself.

  Example:
    iterable = Peeker(iterable)
    first_item = iterable.Peek()
    assert list(iterable)[0] == first_item

  Attributes:
    _iterable: The original iterable.
    _peek: The first item in the iterable, or the iterable itself if its not
      iterable.
    _peek_seen: _peek was already seen by the first next() call.
  """

    def __init__(self, iterable):
        self._iterable = iterable
        self._peek = self._Peek()
        self._peek_seen = False

    def __iter__(self):
        return self

    def _Peek(self):
        """Peeks the first item from the iterable."""
        try:
            return next(self._iterable)
        except TypeError:
            pass
        except StopIteration:
            self._peek_seen = True
            return None
        try:
            return self._iterable.pop(0)
        except (AttributeError, IndexError, KeyError, TypeError):
            pass
        return self._iterable

    def next(self):
        """For Python 2 compatibility."""
        return self.__next__()

    def __next__(self):
        """Returns the next item in the iterable."""
        if not self._peek_seen:
            self._peek_seen = True
            return self._peek
        try:
            return next(self._iterable)
        except TypeError:
            pass
        try:
            return self._iterable.pop(0)
        except AttributeError:
            pass
        except (AttributeError, IndexError, KeyError, TypeError):
            raise StopIteration
        raise StopIteration

    def Peek(self):
        """Returns the first item in the iterable."""
        return self._peek