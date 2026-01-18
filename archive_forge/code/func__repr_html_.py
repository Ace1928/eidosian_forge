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
def _repr_html_(self, max_items=7):
    names = list()
    if len(self) <= max_items:
        names.extend(self.names)
    else:
        half_items = max_items // 2
        for i in range(0, half_items):
            try:
                name = self.names[i]
            except TypeError:
                name = '[no name]'
            names.append(name)
        names.append('...')
        for i in range(-half_items, 0):
            try:
                name = self.names[i]
            except TypeError:
                name = '[no name]'
            names.append(name)
    elements = list()
    for e in self._iter_repr(max_items=max_items):
        if hasattr(e, '_repr_html_'):
            elements.append(tuple(e._iter_formatted()))
        else:
            elements.append(['...'])
    d = {'column_names': names, 'rows': range(len(elements[0]) if len(elements) else 0), 'columns': tuple(range(len(names))), 'nrows': self.nrow, 'ncolumns': self.ncol, 'elements': elements}
    html = self._html_template.render(d)
    return html