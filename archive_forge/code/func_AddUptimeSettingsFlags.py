from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.monitoring import completers
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.command_lib.util.args import repeated
from googlecloudsdk.core.util import times
def AddUptimeSettingsFlags(parser, update=False):
    """Adds uptime check settings flags to the parser."""
    if not update:
        AddUptimeResourceFlags(parser)
    AddUptimeProtocolFlags(parser, update)
    AddUptimeRunFlags(parser, update)
    AddUptimeMatcherFlags(parser)