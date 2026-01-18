from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from typing import Any, Dict, NamedTuple, Optional
from clients import utils as bq_client_utils
from utils import bq_error
def ListCapacityCommitments(client, reference, page_size, page_token):
    """Lists capacity commitments for given project and location.

  Arguments:
    client: The client used to make the request.
    reference: Reference to the project and location.
    page_size: Number of results to show.
    page_token: Token to retrieve the next page of results.

  Returns:
    list of CapacityCommitments objects.
  """
    parent = 'projects/%s/locations/%s' % (reference.projectId, reference.location)
    return client.projects().locations().capacityCommitments().list(parent=parent, pageSize=page_size, pageToken=page_token).execute()