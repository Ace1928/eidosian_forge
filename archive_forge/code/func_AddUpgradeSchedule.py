from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
def AddUpgradeSchedule(parser):
    parser.add_argument('--schedule', required=True, help='\n      Schedule to upgrade a cluster after the request is acknowledged by Google.\n      Support values: IMMEDIATELY.\n      ')