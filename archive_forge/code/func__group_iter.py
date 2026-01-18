import random
import string
from tests.compat import unittest, mock
import boto
def _group_iter(self, iterator, n):
    accumulator = []
    for item in iterator:
        accumulator.append(item)
        if len(accumulator) == n:
            yield accumulator
            accumulator = []
    if len(accumulator) != 0:
        yield accumulator