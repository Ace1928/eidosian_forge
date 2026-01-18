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
def UnicodeIsSupported(cmd_class):
    """Decorator for calliope commands and groups that support unicode.

  Decorate a subclass of base.Command or base.Group with this function, and the
  decorated command or group will not raise the argparse unicode command line
  argument exception.

  Args:
    cmd_class: base._Common, A calliope command or group.

  Returns:
    A modified version of the provided class.
  """
    cmd_class._is_unicode_supported = True
    return cmd_class