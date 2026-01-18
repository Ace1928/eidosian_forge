from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.kms import resource_args as kms_resource_args
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def AddInsightsConfigQueryPlansPerMinute(parser):
    parser.add_argument('--insights-config-query-plans-per-minute', required=False, type=arg_parsers.BoundedInt(lower_bound=0, upper_bound=20), help='Number of query plans to sample every minute.\n        Default value is 5. Allowed range: 0 to 20.')