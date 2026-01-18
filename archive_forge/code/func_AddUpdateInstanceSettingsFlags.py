from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.compute import completers as compute_completers
def AddUpdateInstanceSettingsFlags(parser):
    parser.add_argument('--service-account', help='Email for service account')
    parser.add_argument('--zone', help='Zone for instance settings\n\n', completer=compute_completers.ZonesCompleter, required=True)