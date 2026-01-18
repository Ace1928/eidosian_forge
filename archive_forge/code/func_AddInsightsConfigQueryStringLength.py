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
def AddInsightsConfigQueryStringLength(parser):
    parser.add_argument('--insights-config-query-string-length', required=False, type=arg_parsers.BoundedInt(lower_bound=256, upper_bound=4500), help='Query string length in bytes to be stored by the query insights\n        feature. Default length is 1024 bytes. Allowed range: 256 to 4500\n        bytes.')