from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.recommender import base
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.util.args import common_args
def AddEntityFlagsToParser(parser, entities):
    """Adds argument mutex group of specified entities to parser.

  Args:
      parser: An argparse parser that you can use to add arguments that go on
        the command line after this command.
      entities: The entities to add.
  """
    resource_group = parser.add_mutually_exclusive_group(required=True, help='Resource that is associated with cloud entity type.')
    if base.EntityType.ORGANIZATION in entities:
        resource_group.add_argument('--organization', metavar='ORGANIZATION_ID', help='The Google Cloud organization ID to use for this invocation.')
    if base.EntityType.FOLDER in entities:
        resource_group.add_argument('--folder', metavar='FOLDER_ID', help='The Google Cloud folder ID to use for this invocation.')
    if base.EntityType.BILLING_ACCOUNT in entities:
        resource_group.add_argument('--billing-account', metavar='BILLING_ACCOUNT', help='The Google Cloud billing account ID to use for this invocation.')
    if base.EntityType.PROJECT in entities:
        common_args.ProjectArgument(help_text_to_overwrite='The Google Cloud project ID.').AddToParser(resource_group)