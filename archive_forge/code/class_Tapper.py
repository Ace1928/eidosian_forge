from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
class Tapper(object):
    """Taps an iterable by calling a method for each item and after the last item.

  The returned object is another iterable that is equivalent to the original.
  If the object is not iterable then the first item is the object itself.

  Tappers may be used when it is not efficient or possible to completely drain
  a resource generator before the resources are finally consumed. For example,
  a paged resource may return the first page of resources immediately but have a
  significant delay between subsequent pages. A tapper allows the first page to
  be examined and consumed without waiting for the next page. If the tapper is a
  filter then it can filter and display a page before waiting for the next page.

  Example:
    tap = Tap()
    iterable = Tapper(iterable, tap)
    # The next statement calls tap.Tap(item) for each item and
    # tap.Done() after the last item.
    list(iterable)

  Attributes:
    _iterable: The original iterable.
    _tap: The Tap object.
    _stop: If True then the object is not iterable and it has already been
      returned.
    _injected: True if the previous _call_on_each injected a new item.
    _injected_value: The value to return next.
  """

    def __init__(self, iterable, tap):
        self._iterable = iterable
        self._tap = tap
        self._stop = False
        self._injected = False
        self._injected_value = None

    def __iter__(self):
        return self

    def _NextItem(self):
        """Returns the next item in self._iterable."""
        if self._injected:
            self._injected = False
            return self._injected_value
        try:
            return next(self._iterable)
        except TypeError:
            pass
        except StopIteration:
            self._tap.Done()
            raise
        try:
            return self._iterable.pop(0)
        except (AttributeError, KeyError, TypeError):
            pass
        except IndexError:
            self._tap.Done()
            raise StopIteration
        if self._iterable is None or self._stop:
            self._tap.Done()
            raise StopIteration
        self._stop = True
        return self._iterable

    def next(self):
        """For Python 2 compatibility."""
        return self.__next__()

    def __next__(self):
        """Gets the next item, calls _tap.Tap() on it, and returns it."""
        while True:
            item = self._NextItem()
            inject_or_keep = self._tap.Tap(item)
            if inject_or_keep is None:
                self._tap.Done()
                raise StopIteration
            if isinstance(inject_or_keep, TapInjector):
                if not inject_or_keep.is_replacement:
                    self._injected = True
                    self._injected_value = item
                return inject_or_keep.value
            if inject_or_keep:
                return item