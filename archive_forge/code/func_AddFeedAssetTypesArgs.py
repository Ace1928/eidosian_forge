from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util.apis import yaml_data
from googlecloudsdk.command_lib.util.args import common_args
from googlecloudsdk.command_lib.util.args import resource_args
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def AddFeedAssetTypesArgs(parser):
    parser.add_argument('--asset-types', metavar='ASSET_TYPES', type=arg_parsers.ArgList(), default=[], help='A comma-separated list of types of the assets types to receive updates. For example: `compute.googleapis.com/Disk,compute.googleapis.com/Network`. Regular expressions (https://github.com/google/re2/wiki/Syntax) are also supported. For more information, see: https://cloud.google.com/resource-manager/docs/cloud-asset-inventory/overview')