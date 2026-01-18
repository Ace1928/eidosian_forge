from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.resource_manager import completers
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def GetRecommendFlag(action):
    return base.Argument('--recommend', metavar='BOOLEAN_VALUE', type=arg_parsers.ArgBoolean(), default=False, required=False, help='If true, checks Active Assist recommendation for the risk level of {}, and issues a warning in the prompt. Optional flag is set to false by default. For details see https://cloud.google.com/recommender/docs/change-risk-recommendations'.format(action))