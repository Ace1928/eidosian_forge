from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.compute import batch_helper
from googlecloudsdk.api_lib.compute import path_simplifier
from googlecloudsdk.api_lib.compute import single_request_helper
from googlecloudsdk.api_lib.util import exceptions as http_exceptions
from googlecloudsdk.command_lib.util import time_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
def _RecordUnfinishedOperations(operations, errors):
    """Adds error messages stating that the given operations timed out."""
    pending_resources = [operation.targetLink for operation, _ in operations]
    errors.append((None, 'Did not {action} the following resources within {timeout}s: {links}. These operations may still be underway remotely and may still succeed; use gcloud list and describe commands or https://console.developers.google.com/ to check resource state'.format(action=_HumanFriendlyNameForOpPresentTense(operations[0][0].operationType), timeout=_POLLING_TIMEOUT_SEC, links=', '.join(pending_resources))))