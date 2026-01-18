from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
def GenerateDetailedHelpForAddLabels(resource):
    """Generates the detailed help doc for add-labels command for a resource.

  Args:
    resource: The name of the resource. e.g "instance", "image" or "disk"
  Returns:
    The detailed help doc for the add-labels command.
  """
    return _GenerateDetailedHelpForCommand(resource, _ADD_LABELS_BRIEF_DOC_TEMPLATE, _ADD_LABELS_DESCRIPTION_TEMPLATE, _ADD_LABELS_EXAMPLE_TEMPLATE)