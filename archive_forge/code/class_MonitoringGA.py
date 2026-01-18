from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.GA)
class MonitoringGA(base.Group):
    """Manage Cloud Monitoring dashboards."""
    category = base.MONITORING_CATEGORY
    detailed_help = {'DESCRIPTION': '          Manage Monitoring dashboards.\n\n          More information can be found here:\n              * https://cloud.google.com/monitoring/dashboards/api-dashboard\n      '}

    def Filter(self, context, args):
        base.RequireProjectID(args)
        del context, args