from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.core import properties
def IsManaged():
    return GetPlatform() == PLATFORM_MANAGED