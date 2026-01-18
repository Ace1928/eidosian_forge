from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.util.apis.arg_utils import ChoiceToEnumName
def ParseNodeTemplate(resources, name, project=None, region=None):
    """Parses a node template resource and returns a resource reference.

  Args:
    resources: Resource registry used to parse the node template.
    name: The name of the node template.
    project: The project the node template is associated with.
    region: The region the node temlpate is associated with.

  Returns:
    The parsed node template resource reference.
  """
    return resources.Parse(name, {'project': project, 'region': region}, collection='compute.nodeTemplates')