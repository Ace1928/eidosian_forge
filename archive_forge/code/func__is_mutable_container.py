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
def _is_mutable_container(value):
    """True for mutable containers, which typically need special handling when being copied"""
    return issubclass(type(value), MUTABLE_TYPES)