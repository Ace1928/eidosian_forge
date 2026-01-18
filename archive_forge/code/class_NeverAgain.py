import collections
import fractions
import functools
import heapq
import inspect
import logging
import math
import random
import threading
from concurrent import futures
import futurist
from futurist import _utils as utils
class NeverAgain(Exception):
    """Exception to raise to stop further periodic calls for a function.

    When you want a function never run again you can throw this from
    you periodic function and that will signify to the execution framework
    to remove that function (and never run it again).
    """