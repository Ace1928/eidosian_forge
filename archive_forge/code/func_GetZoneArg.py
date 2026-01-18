from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util import completers
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.concepts import concept_parsers
import ipaddr
def GetZoneArg(help_text='Name of the managed zone whose record sets you want to manage.', hide_short_zone_flag=False):
    """Returns the managed zone arg."""
    if hide_short_zone_flag:
        zone_group = base.ArgumentGroup(required=True)
        zone_group.AddArgument(base.Argument('--zone', completer=ManagedZoneCompleter, help=help_text))
        zone_group.AddArgument(base.Argument('-z', dest='zone', completer=ManagedZoneCompleter, help=help_text, hidden=True))
        return zone_group
    else:
        return base.Argument('--zone', '-z', completer=ManagedZoneCompleter, help=help_text, required=True)