from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.container.gkemulticloud import util as api_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.container.gkemulticloud import constants
from googlecloudsdk.command_lib.projects import util as project_util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import properties
def AddTagsForUpdate(parser, noun):
    """Adds tags related flags for update.

  Args:
    parser: The argparse.parser to add the arguments to.
    noun: The resource type to which the flags are applicable.
  """
    group = parser.add_group('Tags', mutex=True)
    AddTags(group, noun)
    AddClearTags(group, noun)