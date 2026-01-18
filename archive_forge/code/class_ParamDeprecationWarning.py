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
class ParamDeprecationWarning(ParamWarning, DeprecationWarning):
    """Param DeprecationWarning

    Ignored by default except when triggered by code in __main__
    """