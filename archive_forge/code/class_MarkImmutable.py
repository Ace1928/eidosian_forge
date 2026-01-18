import argparse
import builtins
import enum
import importlib
import inspect
import io
import logging
import os
import pickle
import ply.lex
import re
import sys
import textwrap
import types
from operator import attrgetter
from pyomo.common.collections import Sequence, Mapping
from pyomo.common.deprecation import (
from pyomo.common.fileutils import import_file
from pyomo.common.formatting import wrap_reStructuredText
from pyomo.common.modeling import NOTSET
class MarkImmutable(object):
    """
    Mark instances of ConfigValue as immutable.

    Parameters
    ----------
    config_value: ConfigValue
        The ConfigValue instances that should be marked immutable.
        Note that multiple instances of ConfigValue can be passed.

    Examples
    --------
    >>> config = ConfigDict()
    >>> config.declare('a', ConfigValue(default=1, domain=int))
    >>> config.declare('b', ConfigValue(default=1, domain=int))
    >>> locker = MarkImmutable(config.get('a'), config.get('b'))

    Now, config.a and config.b cannot be changed. To make them mutable again,

    >>> locker.release_lock()
    """

    def __init__(self, *args):
        self._targets = args
        self._locked = []
        self.lock()

    def lock(self):
        try:
            for cfg in self._targets:
                if type(cfg) is not ConfigValue:
                    raise ValueError('Only ConfigValue instances can be marked immutable.')
                cfg.__class__ = ImmutableConfigValue
                self._locked.append(cfg)
        except:
            self.release_lock()
            raise

    def release_lock(self):
        for arg in self._locked:
            arg.__class__ = ConfigValue
        self._locked = []

    def __enter__(self):
        if not self._locked:
            self.lock()
        return self

    def __exit__(self, t, v, tb):
        self.release_lock()