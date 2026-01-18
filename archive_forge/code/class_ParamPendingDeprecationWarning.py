import asyncio
import collections
import contextvars
import datetime as dt
import inspect
import functools
import numbers
import os
import re
import sys
import traceback
import warnings
from collections import defaultdict, OrderedDict
from contextlib import contextmanager
from numbers import Real
from textwrap import dedent
from threading import get_ident
from collections import abc
class ParamPendingDeprecationWarning(ParamWarning, PendingDeprecationWarning):
    """Param PendingDeprecationWarning

    This warning type is useful when the warning is not meant to be displayed
    to REPL/notebooks users, as DeprecationWarning are displayed when triggered
    by code in __main__ (__name__ == '__main__' in a REPL).
    """