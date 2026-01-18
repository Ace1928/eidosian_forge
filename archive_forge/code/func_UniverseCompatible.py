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
def UniverseCompatible(cmd_class):
    """Decorator for marking calliope commands and groups available in universes.

  Decorate a subclass of base.Command or base.Group with this function, and the
  decorated command or group will display help text with information about
  command or group is supported in universes.

  Args:
    cmd_class: base._Common, A calliope command or group.

  Returns:
    A modified version of the provided class.
  """
    cmd_class._universe_compatible = True
    return cmd_class