from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core import properties
def GetRevisionResourceArgSpec(group_help):
    """Gets a resource presentation spec for a config manager revision.

  Args:
    group_help: string, the help text for the entire arg group.

  Returns:
    ResourcePresentationSpec for a config manager revision resource argument.
  """
    name = 'REVISION'
    return presentation_specs.ResourcePresentationSpec(name, GetRevisionResourceSpec(), group_help, required=True)