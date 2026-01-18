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
def _state_pop(self_):
    """
        Restore the most recently saved state.

        See _state_push() for more details.
        """
    self = self_.self_or_cls
    if not isinstance(self, Parameterized):
        raise NotImplementedError('_state_pop is not implemented at the class level')
    for pname, p in self.param.objects('existing').items():
        g = self.param.get_value_generator(pname)
        if hasattr(g, '_Dynamic_last'):
            g._Dynamic_last = g._saved_Dynamic_last.pop()
            g._Dynamic_time = g._saved_Dynamic_time.pop()
        elif hasattr(g, '_state_pop') and isinstance(g, Parameterized):
            g._state_pop()