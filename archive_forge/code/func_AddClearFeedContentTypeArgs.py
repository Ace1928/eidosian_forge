from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util.apis import yaml_data
from googlecloudsdk.command_lib.util.args import common_args
from googlecloudsdk.command_lib.util.args import resource_args
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def AddClearFeedContentTypeArgs(parser):
    parser.add_argument('--clear-content-type', action='store_true', help='Clear any existing content type setting on the feed. Content type will be unspecified, no content but the asset name and type will be returned in the feed.')