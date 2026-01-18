import gc
import inspect
import os
import pdb
import random
import sys
import time
import trace
import warnings
from typing import NoReturn, Optional, Type
from twisted import plugin
from twisted.application import app
from twisted.internet import defer
from twisted.python import failure, reflect, usage
from twisted.python.filepath import FilePath
from twisted.python.reflect import namedModule
from twisted.trial import itrial, runner
from twisted.trial._dist.disttrial import DistTrialRunner
from twisted.trial.unittest import TestSuite
def _checkKnownRunOrder(order):
    """
    Check that the given order is a known test running order.

    Does nothing else, since looking up the appropriate callable to sort the
    tests should be done when it actually will be used, as the default argument
    will not be coerced by this function.

    @param order: one of the known orders in C{_runOrders}
    @return: the order unmodified
    """
    if order not in _runOrders:
        raise usage.UsageError('--order must be one of: %s. See --help-orders for details' % (', '.join((repr(order) for order in _runOrders)),))
    return order