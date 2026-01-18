from abc import ABC
import collections
import enum
import functools
import logging
class _EasyOutcome(collections.namedtuple('_EasyOutcome', ['kind', 'return_value', 'exception']), Outcome):
    """A trivial implementation of Outcome."""