import abc
from itertools import compress
import stevedore
from . import command
@staticmethod
def _compress_iterable(iterable, selectors):
    return compress(iterable, selectors)