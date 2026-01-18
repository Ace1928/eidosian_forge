from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.asset import client_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
def AddNamesArgument(parser):
    parser.add_argument('--names', metavar='NAMES', type=arg_parsers.ArgList(min_length=1, max_length=20), required=True, help="        Names refer to a list of\n        [full resource names](https://cloud.google.com/asset-inventory/docs/resource-name-format)\n        of [searchable asset types](https://cloud.google.com/asset-inventory/docs/supported-asset-types).\n        For each batch call, total number of names provided is between 1 and 20.\n\n        The example value is:\n\n          * ```//cloudsql.googleapis.com/projects/{PROJECT_ID}/instances/{INSTANCE}```\n          (e.g. ``//cloudsql.googleapis.com/projects/probe-per-rt-project/instances/instance1'')\n        ")