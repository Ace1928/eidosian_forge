from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
def AddMaintenanceWindowRecurrence(parser):
    parser.add_argument('--maintenance-window-recurrence', help='\n      An RFC 5545 (https://tools.ietf.org/html/rfc5545#section-3.8.5.3)\n        recurrence rule for how the cluster maintenance window recurs. They go\n        on for the span of time between the start and the end time. E.g.\n        FREQ=WEEKLY;BYDAY=SU.\n      ')