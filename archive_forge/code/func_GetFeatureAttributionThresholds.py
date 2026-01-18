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
def GetFeatureAttributionThresholds():
    return base.Argument('--feature-attribution-thresholds', metavar='KEY=VALUE', type=arg_parsers.ArgDict(allow_key_only=True), action=arg_parsers.UpdateAction, help='\nList of feature-attribution score threshold value pairs(Apply for all the\ndeployed models under the endpoint, if you want to specify different thresholds\nfor different deployed model, please use flag --monitoring-config-from-file or\ncall API directly). If only feature name is set, the default threshold value\nwould be 0.3.\n\nFor example: `feature-attribution-thresholds=feat1=0.1,feat2,feat3=0.2`')