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
def OptOutGoogleAuth():
    """Opt-out the command group to use google auth for authentication.

  Call this function in the Filter method of the command group
  to opt-out google-auth.
  """
    properties.VALUES.auth.opt_out_google_auth.Set(True)