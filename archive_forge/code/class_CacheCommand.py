from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import collections
import contextlib
import enum
from functools import wraps  # pylint:disable=g-importing-member
import itertools
import re
import sys
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import display
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.resource import resource_printer
import six
class CacheCommand(six.with_metaclass(abc.ABCMeta, Command)):
    """A command that affects the resource URI cache."""

    def __init__(self, *args, **kwargs):
        super(CacheCommand, self).__init__(*args, **kwargs)
        self._uri_cache_enabled = True