from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util.apis import yaml_data
from googlecloudsdk.command_lib.util.args import common_args
from googlecloudsdk.command_lib.util.args import resource_args
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def AddAssetNamesArgs(parser):
    parser.add_argument('--asset-names', metavar='ASSET_NAMES', required=True, type=arg_parsers.ArgList(), help='A list of full names of the assets to get the history for. For more information, see: https://cloud.google.com/apis/design/resource_names#full_resource_name ')