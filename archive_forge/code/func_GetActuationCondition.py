from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import contextlib
import fnmatch
import io
import json
import os
import re
import signal
from googlecloudsdk.api_lib.container import util
from googlecloudsdk.command_lib.anthos.config.sync.common import exceptions
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core.util import files
def GetActuationCondition(resource_status):
    """Produces a reconciliation condition based on actuation/strategy fields.

    These fields are only present in Config Sync 1.11+.

  Args:
    resource_status (dict): Managed resource status object.

  Returns:
    Condition dict or None.
  """
    actuation = resource_status.get('actuation')
    strategy = resource_status.get('strategy')
    if not actuation or not strategy:
        return None
    statuses_to_report = ['pending', 'skipped', 'failed']
    if str(actuation).lower() in statuses_to_report:
        return {'message': 'Resource pending {}'.format(strategy)}
    return None