from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import os
import zipfile
from googlecloudsdk.api_lib import apigee
from googlecloudsdk.command_lib.apigee import errors
from googlecloudsdk.core import log
from googlecloudsdk.core import requests
from googlecloudsdk.core.resource import resource_projector
from googlecloudsdk.core.util import archive
from googlecloudsdk.core.util import files
from six.moves import urllib
def _StatusFromOperation(self, op):
    """Gathers given an LRO, determines the associated archive status.

    Args:
      op: An Apigee LRO

    Returns:
      A dict in the format of
        {"status": "{status}", "error": "{error if present on LRO}"}.
    """
    status = {}
    try:
        is_done = self._lro_helper.IsDone(op)
        if is_done:
            status['status'] = self._deployed_status
        else:
            status['status'] = self._inprogress_status
    except errors.RequestError:
        status['status'] = self._failed_status
        status['error'] = op['error']['message']
    return status