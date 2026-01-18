from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
class Conditions(base.Group):
    """Manage Cloud Monitoring alerting policy conditions."""
    detailed_help = {'DESCRIPTION': '          Manage Monitoring alerting policies conditions.\n\n          More information can be found here:\n          https://cloud.google.com/monitoring/api/v3/\n      '}