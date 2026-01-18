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
def EnableUserProjectQuotaWithFallback():
    """Tries the current project and fall back to the legacy mode.

  The project in core/project will be used to populate the quota project header.
  It should be used in command group's Filter function so that commands in the
  group will send the current project (core/project) in the quota project
  header. If the user does not have the permission to use the project,
  we will retry the request after removing the quota project header.

  See the docstring of DisableUserProjectQuota for more information.
  """
    _SetUserProjectQuotaFallback(properties.VALUES.billing.CURRENT_PROJECT_WITH_FALLBACK)