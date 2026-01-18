from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.recommender import base
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.util.args import common_args
def GetInsightTypeConfigName(args):
    """Returns the resource name for the insight type config."""
    return GetInsightTypeName(args) + '/config'