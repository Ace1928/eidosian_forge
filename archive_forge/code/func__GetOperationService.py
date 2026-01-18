from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.firestore import api_utils
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.core import resources
def _GetOperationService():
    """Returns the service for interacting with the Operations service."""
    return api_utils.GetClient().projects_databases_operations