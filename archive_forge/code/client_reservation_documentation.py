from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from typing import Any, Dict, NamedTuple, Optional
from clients import utils as bq_client_utils
from utils import bq_error
Searches reservations assignments for given assignee.

  Arguments:
    client: The client used to make the request.
    location: location of interest.
    job_type: type of job to be queried.
    assignee_type: Type of assignees for the reservation assignment.
    assignee_id: Project/folder/organization ID, to which the reservation is
      assigned.

  Returns:
    ReservationAssignment object if it exists.

  Raises:
    bq_error.BigqueryError: If required parameters are not passed in or
      reservation assignment not found.
  