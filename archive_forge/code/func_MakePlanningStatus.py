from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute.reservations import util as reservation_util
from googlecloudsdk.core.util import times
def MakePlanningStatus(messages, planning_status):
    """Constructs the planning status enum value."""
    if planning_status:
        if planning_status == 'SUBMITTED':
            return messages.FutureReservation.PlanningStatusValueValuesEnum.SUBMITTED
    return None