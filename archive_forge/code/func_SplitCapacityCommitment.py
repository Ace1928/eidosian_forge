from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from typing import Any, Dict, NamedTuple, Optional
from clients import utils as bq_client_utils
from utils import bq_error
def SplitCapacityCommitment(client, reference, slots):
    """Splits a capacity commitment with the given reference into two.

  Arguments:
    client: The client used to make the request.
    reference: Capacity commitment to split.
    slots: Number of slots in the first capacity commitment after the split.

  Returns:
    List of capacity commitment objects after the split.

  Raises:
    bq_error.BigqueryError: if capacity commitment cannot be updated.
  """
    if slots is None:
        raise bq_error.BigqueryError('Please specify slots for the split.')
    body = {'slotCount': slots}
    response = client.projects().locations().capacityCommitments().split(name=reference.path(), body=body).execute()
    if 'first' not in response or 'second' not in response:
        raise bq_error.BigqueryError('internal error')
    return [response['first'], response['second']]