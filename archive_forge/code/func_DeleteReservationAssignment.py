from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from typing import Any, Dict, NamedTuple, Optional
from clients import utils as bq_client_utils
from utils import bq_error
def DeleteReservationAssignment(client, reference):
    """Deletes given reservation assignment.

  Arguments:
    client: The client used to make the request.
    reference: Reference to the reservation assignment.
  """
    client.projects().locations().reservations().assignments().delete(name=reference.path()).execute()