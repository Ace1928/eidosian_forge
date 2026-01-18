from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.endpoints import services_util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.endpoints import arg_parsers
from googlecloudsdk.command_lib.endpoints import common_flags
from googlecloudsdk.core import resources
def _GetServiceName():
    return arg_parsers.GetServiceNameFromArg(args.MakeGetOrRaise('--service')())