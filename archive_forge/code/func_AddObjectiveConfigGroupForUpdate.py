from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import sys
import textwrap
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.ai import constants
from googlecloudsdk.command_lib.ai import errors
from googlecloudsdk.command_lib.ai import region_util
from googlecloudsdk.command_lib.iam import iam_util as core_iam_util
from googlecloudsdk.command_lib.kms import resource_args as kms_resource_args
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def AddObjectiveConfigGroupForUpdate(parser, required=False):
    """Add model monitoring objective config related flags to the parser for Update API.
  """
    objective_config_group = parser.add_mutually_exclusive_group(required=required)
    thresholds_group = objective_config_group.add_group(mutex=False)
    GetFeatureThresholds().AddToParser(thresholds_group)
    GetFeatureAttributionThresholds().AddToParser(thresholds_group)
    GetMonitoringConfigFromFile().AddToParser(objective_config_group)