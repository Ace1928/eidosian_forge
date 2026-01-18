from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util.apis import yaml_data
from googlecloudsdk.command_lib.util.args import common_args
from googlecloudsdk.command_lib.util.args import resource_args
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def AddAssetTypesArgs(parser):
    parser.add_argument('--asset-types', metavar='ASSET_TYPES', type=arg_parsers.ArgList(), default=[], help='A list of asset types (i.e., "compute.googleapis.com/Disk") to take a snapshot. If specified and non-empty, only assets matching the specified types will be returned. See http://cloud.google.com/asset-inventory/docs/supported-asset-types for supported asset types.')