from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.edge_cloud.container import util
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.run import flags
def RequestWithNewMaintenanceExclusion(req, messages, args):
    """Returns an update request with a new maintenance exclusion window with id, start time, and end time specified from args.

  Args:
    req: API request to be issued.
    messages: message module of edgecontainer cluster.
    args: command line arguments.

  Returns:
    modified request
  """
    if req.cluster.maintenancePolicy is None:
        req.cluster.maintenancePolicy = messages.MaintenancePolicy()
    if req.cluster.maintenancePolicy.maintenanceExclusions is None:
        req.cluster.maintenancePolicy.maintenanceExclusions = []
    req.cluster.maintenancePolicy.maintenanceExclusions.append(messages.MaintenanceExclusionWindow(id=args.add_maintenance_exclusion_name, window=messages.TimeWindow(startTime=args.add_maintenance_exclusion_start, endTime=args.add_maintenance_exclusion_end)))
    return req