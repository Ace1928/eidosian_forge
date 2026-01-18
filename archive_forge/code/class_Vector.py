import abc
import collections.abc
from rpy2.robjects.robject import RObjectMixin
import rpy2.rinterface as rinterface
from rpy2.rinterface_lib import sexp
from . import conversion
import rpy2.rlike.container as rlc
import datetime
import copy
import itertools
import math
import os
import jinja2  # type: ignore
import time
import tzlocal
from time import struct_time, mktime
import typing
import warnings
from rpy2.rinterface import (Sexp, ListSexpVector, StrSexpVector,
class Vector(RObjectMixin):
    """Vector(seq) -> Vector.

    The parameter 'seq' can be an instance inheriting from
    rinterface.SexpVector, or an arbitrary Python object.
    In the later case, a conversion will be attempted using
    conversion.get_conversion().py2rpy().

    R vector-like object. Items can be accessed with:

    - the method "__getitem__" ("[" operator)

    - the delegators rx or rx2
    """
    _html_template = jinja2.Template('\n        <span>{{ classname }} with {{ nelements }} elements.</span>\n        <table>\n        <tbody>\n          <tr>\n          {% for elt in elements %}\n            <td>\n            {{ elt }}\n            </td>\n          {% endfor %}\n          </tr>\n        </tbody>\n        </table>\n        ')

    def _add_rops(self):
        self.ro = VectorOperationsDelegator(self)
        self.rx = ExtractDelegator(self)
        self.rx2 = DoubleExtractDelegator(self)

    def __add__(self, x):
        cv = conversion.get_conversion()
        res = baseenv_ri.find('c')(self, cv.py2rpy(x))
        res = cv.rpy2py(res)
        return res

    def __getitem__(self, i):
        res = super().__getitem__(i)
        if isinstance(res, Sexp):
            res = conversion.get_conversion().rpy2py(res)
        return res

    def __setitem__(self, i, value):
        value = conversion.get_conversion().py2rpy(value)
        super().__setitem__(i, value)

    @property
    def names(self):
        """Names for the items in the vector."""
        res = super().names
        res = conversion.get_conversion().rpy2py(res)
        return res

    @names.setter
    def names(self, value):
        res = globalenv_ri.find('names<-')(self, conversion.get_conversion().py2rpy(value))
        self.__sexp__ = res.__sexp__

    def items(self):
        """ iterator on names and values """
        if super().names.rsame(rinterface.NULL):
            it_names = itertools.cycle((None,))
        else:
            it_names = iter(self.names)
        it_self = iter(self)
        for v, k in zip(it_self, it_names):
            yield (k, v)

    def sample(self: collections.abc.Sized, n: int, replace: bool=False, probabilities: typing.Optional[collections.abc.Sized]=None):
        """ Draw a random sample of size n from the vector.

        If 'replace' is True, the sampling is done with replacement.
        The optional argument 'probabilities' can indicate sampling
        probabilities."""
        assert isinstance(n, int)
        assert isinstance(replace, bool)
        if probabilities is not None:
            if len(probabilities) != len(self):
                raise ValueError('The sequence of probabilities must match the length of the vector.')
            if not isinstance(probabilities, rinterface.FloatSexpVector):
                probabilities = FloatVector(probabilities)
        res = _sample(self, IntVector((n,)), replace=BoolVector((replace,)), prob=probabilities)
        res = conversion.rpy2py(res)
        return res

    def repr_format_elt(self, elt, max_width=12):
        max_width = int(max_width)
        if elt in (NA_Real, NA_Integer, NA_Character, NA_Logical):
            res = repr(elt)
        elif isinstance(elt, int):
            res = '%8i' % elt
        elif isinstance(elt, float):
            res = '%8f' % elt
        else:
            if isinstance(elt, str):
                elt = repr(elt)
            else:
                elt = type(elt).__name__
            if len(elt) < max_width:
                res = elt
            else:
                res = '%s...' % str(elt[:max_width - 3])
        return res

    def _iter_formatted(self, max_items=9):
        format_elt = self.repr_format_elt
        ln = len(self)
        half_items = max_items // 2
        if ln == 0:
            return
        elif ln < max_items:
            for elt in conversion.noconversion(self):
                yield format_elt(elt, max_width=math.floor(52 / ln))
        else:
            for elt in conversion.noconversion(self)[:half_items]:
                yield format_elt(elt)
            yield '...'
            for elt in conversion.noconversion(self)[-half_items:]:
                yield format_elt(elt)

    def __repr_content__(self):
        return ''.join(('[', ', '.join(self._iter_formatted()), ']'))

    def __repr__(self):
        return super().__repr__() + os.linesep + self.__repr_content__()

    def _repr_html_(self, max_items=7):
        d = {'elements': self._iter_formatted(max_items=max_items), 'classname': type(self).__name__, 'nelements': len(self)}
        html = self._html_template.render(d)
        return html