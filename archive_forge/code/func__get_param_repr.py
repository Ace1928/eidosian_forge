import asyncio
import copy
import datetime as dt
import html
import inspect
import logging
import numbers
import operator
import random
import re
import types
import typing
import warnings
from collections import defaultdict, namedtuple, OrderedDict
from functools import partial, wraps, reduce
from html import escape
from itertools import chain
from operator import itemgetter, attrgetter
from types import FunctionType, MethodType
from contextlib import contextmanager
from logging import DEBUG, INFO, WARNING, ERROR, CRITICAL
from ._utils import (
from inspect import getfullargspec
def _get_param_repr(key, val, p, vallen=30, doclen=40):
    """HTML representation for a single Parameter object and its value"""
    if isinstance(val, Parameterized) or (type(val) is type and issubclass(val, Parameterized)):
        value = val.param._repr_html_(open=False)
    elif hasattr(val, '_repr_html_'):
        value = val._repr_html_()
    else:
        value = truncate(repr(val), vallen)
    if hasattr(p, 'bounds'):
        if p.bounds is None:
            range_ = ''
        elif hasattr(p, 'inclusive_bounds'):
            bl, bu = p.bounds
            il, iu = p.inclusive_bounds
            lb = '' if bl is None else ('>=' if il else '>') + str(bl)
            ub = '' if bu is None else ('<=' if iu else '<') + str(bu)
            range_ = lb + (', ' if lb and bu else '') + ub
        else:
            range_ = repr(p.bounds)
    elif hasattr(p, 'objects') and p.objects:
        range_ = ', '.join(list(map(repr, p.objects)))
    elif hasattr(p, 'class_'):
        if isinstance(p.class_, tuple):
            range_ = ' | '.join((kls.__name__ for kls in p.class_))
        else:
            range_ = p.class_.__name__
    elif hasattr(p, 'regex') and p.regex is not None:
        range_ = f'regex({p.regex})'
    else:
        range_ = ''
    if p.readonly:
        range_ = ' '.join((s for s in ['<i>read-only</i>', range_] if s))
    elif p.constant:
        range_ = ' '.join((s for s in ['<i>constant</i>', range_] if s))
    if getattr(p, 'allow_None', False):
        range_ = ' '.join((s for s in ['<i>nullable</i>', range_] if s))
    tooltip = f' class="param-doc-tooltip" data-tooltip="{escape(p.doc.strip())}"' if p.doc else ''
    return f'<tr>  <td><p style="margin-bottom: 0px;"{tooltip}>{key}</p></td>  <td style="max-width: 200px; text-align:left;">{value}</td>  <td style="text-align:left;">{p.__class__.__name__}</td>  <td style="max-width: 300px;">{range_}</td></tr>\n'