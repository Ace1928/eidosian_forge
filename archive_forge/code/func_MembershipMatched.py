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
def MembershipMatched(membership, target_membership):
    """Check if the current membership matches the specified memberships.

  Args:
    membership: string The current membership.
    target_membership: string The specified memberships.

  Returns:
    Returns True if matching; False otherwise.
  """
    if not target_membership:
        return True
    if target_membership and '*' in target_membership:
        return fnmatch.fnmatch(membership, target_membership)
    else:
        members = target_membership.split(',')
        for m in members:
            if m == membership:
                return True
        return False