from collections.abc import Iterable, Sequence
from contextlib import ExitStack
import functools
import inspect
import logging
from numbers import Real
from operator import attrgetter
import types
import numpy as np
import matplotlib as mpl
from matplotlib import _api, cbook, _docstring, offsetbox
import matplotlib.artist as martist
import matplotlib.axis as maxis
from matplotlib.cbook import _OrderedSet, _check_1d, index_of
import matplotlib.collections as mcoll
import matplotlib.colors as mcolors
import matplotlib.font_manager as font_manager
from matplotlib.gridspec import SubplotSpec
import matplotlib.image as mimage
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.rcsetup import cycler, validate_axisbelow
import matplotlib.spines as mspines
import matplotlib.table as mtable
import matplotlib.text as mtext
import matplotlib.ticker as mticker
import matplotlib.transforms as mtransforms
class _axis_method_wrapper:
    """
    Helper to generate Axes methods wrapping Axis methods.

    After ::

        get_foo = _axis_method_wrapper("xaxis", "get_bar")

    (in the body of a class) ``get_foo`` is a method that forwards it arguments
    to the ``get_bar`` method of the ``xaxis`` attribute, and gets its
    signature and docstring from ``Axis.get_bar``.

    The docstring of ``get_foo`` is built by replacing "this Axis" by "the
    {attr_name}" (i.e., "the xaxis", "the yaxis") in the wrapped method's
    dedented docstring; additional replacements can be given in *doc_sub*.
    """

    def __init__(self, attr_name, method_name, *, doc_sub=None):
        self.attr_name = attr_name
        self.method_name = method_name
        doc = inspect.getdoc(getattr(maxis.Axis, method_name))
        self._missing_subs = []
        if doc:
            doc_sub = {'this Axis': f'the {self.attr_name}', **(doc_sub or {})}
            for k, v in doc_sub.items():
                if k not in doc:
                    self._missing_subs.append(k)
                doc = doc.replace(k, v)
        self.__doc__ = doc

    def __set_name__(self, owner, name):
        get_method = attrgetter(f'{self.attr_name}.{self.method_name}')

        def wrapper(self, *args, **kwargs):
            return get_method(self)(*args, **kwargs)
        wrapper.__module__ = owner.__module__
        wrapper.__name__ = name
        wrapper.__qualname__ = f'{owner.__qualname__}.{name}'
        wrapper.__doc__ = self.__doc__
        wrapper.__signature__ = inspect.signature(getattr(maxis.Axis, self.method_name))
        if self._missing_subs:
            raise ValueError('The definition of {} expected that the docstring of Axis.{} contains {!r} as substrings'.format(wrapper.__qualname__, self.method_name, ', '.join(map(repr, self._missing_subs))))
        setattr(owner, name, wrapper)