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
def _SleepMs(time_to_wait_ms):
    time.sleep(time_to_wait_ms / 1000.0)