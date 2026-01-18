from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from typing import Any, Dict, NamedTuple, Optional
from clients import utils as bq_client_utils
from utils import bq_error
def UpdateReservation(client, api_version: str, reference, slots, ignore_idle_slots, target_job_concurrency: Optional[int], autoscale_max_slots):
    """Updates a reservation with the given reservation reference.

  Arguments:
    client: The client used to make the request.
    api_version: The api version to make the request against.
    reference: Reservation to update.
    slots: Number of slots allocated to this reservation subtree.
    ignore_idle_slots: Specifies whether queries should ignore idle slots from
      other reservations.
    target_job_concurrency: Job concurrency target.
    autoscale_max_slots: Number of slots to be scaled when needed.

  Returns:
    Reservation object that was updated.

  Raises:
    bq_error.BigqueryError: if autoscale_max_slots is used with other
      version.
  """
    reservation, update_mask = GetParamsForUpdateReservation(api_version, slots, ignore_idle_slots, target_job_concurrency, autoscale_max_slots)
    return client.projects().locations().reservations().patch(name=reference.path(), updateMask=update_mask, body=reservation).execute()