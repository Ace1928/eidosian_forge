from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import functools
import itertools
import math
import random
import sys
import time
from googlecloudsdk.core import exceptions
def ShouldRetryFunc(try_func_result, state):
    exc_info = try_func_result[1]
    if exc_info is None:
        return False
    return should_retry_if(exc_info[0], exc_info[1], exc_info[2], state)