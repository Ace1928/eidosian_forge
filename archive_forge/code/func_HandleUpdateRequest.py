from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.container.fleet.scopes.rollout_sequencing import base
from googlecloudsdk.core import log
def HandleUpdateRequest(ref, args):
    cmd = base.UpdateCommand(args)
    response = cmd.hubclient.messages.Scope(name=ref.RelativeName())
    return response