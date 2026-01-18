from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.instances import flags as instance_flags
from googlecloudsdk.command_lib.compute.reservations import flags as reservation_flags
from googlecloudsdk.command_lib.compute.reservations import resource_args
from googlecloudsdk.command_lib.util.apis import arg_utils
def AddExistingReservationFlag(parser):
    """Add --existing-reservation argument to parser."""
    help_text = '\n  Details of the existing on-demand reservation or auto-created future\n  reservation that you want to attach to your commitment. Specify a new instance\n  of this flag for every existing reservation that you want to attach. The\n  reservations must be in the same region as the commitment.\n  *name*::: The name of the reservation.\n  *zone*::: The zone of the reservation.\n  For example, to attach an existing reservation named reservation-name in the\n  zone reservation-zone, use the following text:\n  --existing-reservation=name=reservation-name,zone=reservation-zone\n  '
    return parser.add_argument('--existing-reservation', type=arg_parsers.ArgDict(spec={'name': str, 'zone': str}, required_keys=['name', 'zone']), action='append', help=help_text)