from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.firestore import api_utils
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.core import resources
def CancelOperation(name):
    """Cancels the Operation with the given name."""
    return _GetOperationService().Cancel(api_utils.GetMessages().FirestoreProjectsDatabasesOperationsCancelRequest(name=name))