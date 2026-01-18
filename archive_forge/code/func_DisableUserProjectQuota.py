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
def DisableUserProjectQuota():
    """Disable the quota project header.

   This function will set the value for properties.VALUES.billing.quota_project
   which is used to decide if we want to send the quota project header
   x-goog-user-project and what value to put in the header. Gcloud's property
   has multiple layers of fallbacks when resolving its value. Specifically for
   quota_project property:

   L1: invocation stack populated by parsing --billing-project.
   L2: its env variable CLOUDSDK_BILLING_QUOTA_PROJECT
   L3: user configuration or installation configuration from gcloud config set.
   L4: value provided by its fallbacks if exists.
   L5: default value

  This function sets the value at L4 (fallbacks). It should be used in command
  group's Filter function so that the command group will work in LEGACY mode
  (DO NOT send the quota project header). It sets at L4 because:

  1. L1-L3 are user settings we want to honor. This func
     cannot operate in L1-L3 because it will mix with user settings.
     Whether the setting is from user is an important information when we decide
     how override works.
  2. L4 can override the default value (L5).
  """
    _SetUserProjectQuotaFallback(properties.VALUES.billing.LEGACY)