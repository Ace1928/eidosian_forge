from __future__ import division  # Many analytical derivatives depend on this
from builtins import str, next, map, zip, range, object
import math
from math import sqrt, log, isnan, isinf  # Optimization: no attribute look-up
import re
import sys
import copy
import warnings
import itertools
import inspect
import numbers
import collections
class IndexableIter(object):
    """
    Iterable whose values can also be accessed through indexing.

    The input iterable values are cached.

    Some attributes:

    iterable -- iterable used for returning the elements one by one.

    returned_elements -- list with the elements directly accessible.
    through indexing. Additional elements are obtained from self.iterable.

    none_converter -- function that takes an index and returns the
    value to be returned when None is obtained form the iterable
    (instead of None).
    """

    def __init__(self, iterable, none_converter=lambda index: None):
        """
        iterable -- iterable whose values will be returned.

        none_converter -- function applied to None returned
        values. The value that replaces None is none_converter(index),
        where index is the index of the element.
        """
        self.iterable = iterable
        self.returned_elements = []
        self.none_converter = none_converter

    def __getitem__(self, index):
        returned_elements = self.returned_elements
        try:
            return returned_elements[index]
        except IndexError:
            for pos in range(len(returned_elements), index + 1):
                value = next(self.iterable)
                if value is None:
                    value = self.none_converter(pos)
                returned_elements.append(value)
            return returned_elements[index]

    def __str__(self):
        return '<%s: [%s...]>' % (self.__class__.__name__, ', '.join(map(str, self.returned_elements)))