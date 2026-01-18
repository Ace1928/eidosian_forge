from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
def GenerateDetailedHelpForRemoveLabels(resource):
    """Generates the detailed help doc for remove-labels command for a resource.

  Args:
    resource: The name of the resource. e.g "instance", "image" or "disk"
  Returns:
    The detailed help doc for the remove-labels command.
  """
    return _GenerateDetailedHelpForCommand(resource, _REMOVE_LABELS_BRIEF_DOC_TEMPLATE, _REMOVE_LABELS_DESCRIPTION_TEMPLATE, _REMOVE_LABELS_EXAMPLE_TEMPLATE)