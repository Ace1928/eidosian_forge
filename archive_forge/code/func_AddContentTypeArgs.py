from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util.apis import yaml_data
from googlecloudsdk.command_lib.util.args import common_args
from googlecloudsdk.command_lib.util.args import resource_args
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def AddContentTypeArgs(parser, required):
    """--content-type argument for asset export and get-history."""
    if required:
        help_text = 'Asset content type.'
    else:
        help_text = 'Asset content type. If specified, only content matching the specified type will be returned. Otherwise, no content but the asset name will be returned.'
    help_text += ' Specifying `resource` will export resource metadata, specifying `iam-policy` will export the IAM policy for each child asset, specifying `org-policy` will export the Org Policy set on child assets, specifying `access-policy` will export the Access Policy set on child assets, specifying `os-inventory` will export the OS inventory of VM instances, and specifying `relationship` will export relationships of the assets.'
    parser.add_argument('--content-type', required=required, choices=['resource', 'iam-policy', 'org-policy', 'access-policy', 'os-inventory', 'relationship'], help=help_text)