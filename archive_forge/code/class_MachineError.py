import inspect
import itertools
import logging
import warnings
from collections import OrderedDict, defaultdict, deque
from functools import partial
from six import string_types
class MachineError(Exception):
    """ MachineError is used for issues related to state transitions and current states.
    For instance, it is raised for invalid transitions or machine configuration issues.
    """

    def __init__(self, value):
        super(MachineError, self).__init__(value)
        self.value = value

    def __str__(self):
        return repr(self.value)