from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.alloydb import flags
from googlecloudsdk.core import properties
def _ConstructDenyPeriods(alloydb_messages, args, update=False):
    """Returns the deny periods based on args."""
    if update and args.remove_deny_maintenance_period:
        return []
    deny_period = alloydb_messages.DenyMaintenancePeriod()
    deny_period.startDate = args.deny_maintenance_period_start_date
    deny_period.endDate = args.deny_maintenance_period_end_date
    deny_period.time = args.deny_maintenance_period_time
    return [deny_period]