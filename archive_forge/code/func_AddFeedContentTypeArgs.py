from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util.apis import yaml_data
from googlecloudsdk.command_lib.util.args import common_args
from googlecloudsdk.command_lib.util.args import resource_args
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def AddFeedContentTypeArgs(parser):
    help_text = 'Asset content type. If not specified, no content but the asset name and type will be returned in the feed. For more information, see https://cloud.google.com/resource-manager/docs/cloud-asset-inventory/overview#asset_content_type'
    FeedContentTypeArgs(parser, help_text)