import atexit
import operator
import os
import sys
import threading
import time
import traceback as _traceback
import warnings
import subprocess
import functools
from more_itertools import always_iterable
class _StateEnum(object):

    class State(object):
        name = None

        def __repr__(self):
            return 'states.%s' % self.name

    def __setattr__(self, key, value):
        if isinstance(value, self.State):
            value.name = key
        object.__setattr__(self, key, value)