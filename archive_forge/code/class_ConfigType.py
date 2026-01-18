from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import enum
import json
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import files
import six
class ConfigType(enum.Enum):
    WORKLOAD_IDENTITY_POOLS = 1
    WORKFORCE_POOLS = 2