from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import re
from apitools.base.py import encoding
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.endpoints import exceptions
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core import yaml
from googlecloudsdk.core.resource import resource_printer
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import retry
import six
def GetActiveServiceConfigIdsFromRollout(rollout):
    """Get the active service config IDs from a Rollout message.

  Args:
    rollout: The rollout message to inspect.

  Returns:
    A list of active service config IDs as indicated in the rollout message.
  """
    if rollout and rollout.trafficPercentStrategy:
        return [p.key for p in rollout.trafficPercentStrategy.percentages.additionalProperties]
    else:
        return []