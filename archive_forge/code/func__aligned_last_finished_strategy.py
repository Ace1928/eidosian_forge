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
def _aligned_last_finished_strategy(cb, started_at, finished_at, metrics):
    how_often = cb._periodic_spacing
    aligned_finished_at = finished_at - math.fmod(finished_at, how_often)
    return aligned_finished_at + how_often