from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from typing import Any, Dict, NamedTuple, Optional
from clients import utils as bq_client_utils
from utils import bq_error
def UpdateReservationAssignment(client, reference, priority):
    """Updates reservation assignment.

  Arguments:
    client: The client used to make the request.
    reference: Reference to the reservation assignment.
    priority: Default job priority for this assignment.

  Returns:
    Reservation assignment object that was updated.

  Raises:
    bq_error.BigqueryError: if assignment cannot be updated.
  """
    reservation_assignment = {}
    update_mask = ''
    if priority is not None:
        if not priority:
            priority = 'JOB_PRIORITY_UNSPECIFIED'
        reservation_assignment['priority'] = priority
        update_mask += 'priority,'
    return client.projects().locations().reservations().assignments().patch(name=reference.path(), updateMask=update_mask, body=reservation_assignment).execute()