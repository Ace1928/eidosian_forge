from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
def AddNetworkArgsForResume(parser):
    help_text_override = "    Set to the network that was originally used creating the suspended Cloud TPU\n    and Compute Engine VM. (It defaults to using the 'default' network.)\n    "
    return AddNetworkArgs(parser, help_text_override)