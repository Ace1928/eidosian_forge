from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from typing import Any, Dict, NamedTuple, Optional
from clients import utils as bq_client_utils
from utils import bq_error
def CreateReservationAssignment(client, reference, job_type, priority, assignee_type, assignee_id):
    """Creates a reservation assignment for a given project/folder/organization.

  Arguments:
    client: The client used to make the request.
    reference: Reference to the project reservation is assigned. Location must
      be the same location as the reservation.
    job_type: Type of jobs for this assignment.
    priority: Default job priority for this assignment.
    assignee_type: Type of assignees for the reservation assignment.
    assignee_id: Project/folder/organization ID, to which the reservation is
      assigned.

  Returns:
    ReservationAssignment object that was created.

  Raises:
    bq_error.BigqueryError: if assignment cannot be created.
  """
    reservation_assignment = {}
    if not job_type:
        raise bq_error.BigqueryError('job_type not specified.')
    reservation_assignment['job_type'] = job_type
    if priority:
        reservation_assignment['priority'] = priority
    if not assignee_type:
        raise bq_error.BigqueryError('assignee_type not specified.')
    if not assignee_id:
        raise bq_error.BigqueryError('assignee_id not specified.')
    reservation_assignment['assignee'] = '%ss/%s' % (assignee_type.lower(), assignee_id)
    return client.projects().locations().reservations().assignments().create(parent=reference.path(), body=reservation_assignment).execute()