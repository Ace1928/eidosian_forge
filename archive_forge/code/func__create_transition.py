import inspect
import itertools
import logging
import warnings
from collections import OrderedDict, defaultdict, deque
from functools import partial
from six import string_types
@classmethod
def _create_transition(cls, *args, **kwargs):
    return cls.transition_cls(*args, **kwargs)