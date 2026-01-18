from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.compute.networks import flags as compute_network_flags
from googlecloudsdk.command_lib.compute.networks.subnets import flags as compute_subnet_flags
from googlecloudsdk.command_lib.kms import resource_args as kms_resource_args
from googlecloudsdk.command_lib.notebooks import completers
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import properties
def AddDeleteEnvironmentFlags(api_version, parser):
    GetEnvironmentResourceArg(api_version, 'User-defined unique name of this environment. The environment name must be 1 to 63 characters long and contain only lowercase letters, numeric characters, and dashes. The first character must be a lowercaseletter and the last character cannot be a dash.').AddToParser(parser)
    base.ASYNC_FLAG.AddToParser(parser)