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
class _ClassPrivate:
    """
    parameters_state: dict
        Dict holding some transient states
    disable_instance_params: bool
        Whether to disable instance parameters
    renamed: bool
        Whethe the class has been renamed by a super class
    params: dict
        Dict of parameter_name:parameter
    """
    __slots__ = ['parameters_state', 'disable_instance_params', 'renamed', 'params', 'initialized', 'signature', 'explicit_no_refs']

    def __init__(self, parameters_state=None, disable_instance_params=False, explicit_no_refs=None, renamed=False, params=None):
        if parameters_state is None:
            parameters_state = {'BATCH_WATCH': False, 'TRIGGER': False, 'events': [], 'watchers': []}
        self.parameters_state = parameters_state
        self.disable_instance_params = disable_instance_params
        self.renamed = renamed
        self.params = {} if params is None else params
        self.initialized = False
        self.signature = None
        self.explicit_no_refs = [] if explicit_no_refs is None else explicit_no_refs

    def __getstate__(self):
        return {slot: getattr(self, slot) for slot in self.__slots__}

    def __setstate__(self, state):
        for k, v in state.items():
            setattr(self, k, v)