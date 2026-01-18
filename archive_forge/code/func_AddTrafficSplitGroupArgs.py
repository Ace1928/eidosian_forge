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
def AddTrafficSplitGroupArgs(parser):
    """Add arguments for traffic split."""
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--traffic-split', metavar='DEPLOYED_MODEL_ID=VALUE', type=arg_parsers.ArgDict(value_type=int), action=arg_parsers.UpdateAction, help='List of pairs of deployed model id and value to set as traffic split.')
    group.add_argument('--clear-traffic-split', action='store_true', help='Clears the traffic split map. If the map is empty, the endpoint is to not accept any traffic at the moment.')