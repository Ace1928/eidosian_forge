from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from googlecloudsdk.calliope import base
def AddNotificationConfigPositionalArgument(parser):
    """Add Notification Config as a positional argument."""
    parser.add_argument('NOTIFICATIONCONFIGID', metavar='NOTIFICATION_CONFIG_ID', help='      The ID of the notification config. Formatted as\n      "organizations/123/notificationConfigs/456" or just "456".\n      ')
    return parser