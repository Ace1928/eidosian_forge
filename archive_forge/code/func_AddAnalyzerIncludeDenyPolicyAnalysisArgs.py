from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util.apis import yaml_data
from googlecloudsdk.command_lib.util.args import common_args
from googlecloudsdk.command_lib.util.args import resource_args
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def AddAnalyzerIncludeDenyPolicyAnalysisArgs(parser):
    """Adds include deny policy analysis arg into options.

  Args:
    parser: the option group.
  """
    parser.add_argument('--include-deny-policy-analysis', action='store_true', help='If true, the response will include analysis for deny policies.This is a very expensive operation, because many derived queries will be executed.')
    parser.set_defaults(include_deny_policy_analysis=False)