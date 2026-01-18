from __future__ import absolute_import
from future.utils import PY2
from itertools import islice
from future.backports.misc import count   # with step parameter on Py2.6
class range_iterator(Iterator):
    """An iterator for a :class:`range`.
    """

    def __init__(self, range_):
        self._stepper = islice(count(range_.start, range_.step), len(range_))

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._stepper)

    def next(self):
        return next(self._stepper)