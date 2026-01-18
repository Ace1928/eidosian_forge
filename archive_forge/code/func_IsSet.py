from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.calliope import parser_arguments
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.container.gkeonprem import flags
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def IsSet(kwargs):
    """Returns True if any of the kwargs is set to not None value.

  The added condition handles the case when user specified boolean False
  for the given args, and it should return True, which does not work with
  normal Python identity comparison.

  Args:
    kwargs: dict, a mapping from proto field to its respective constructor
      function.

  Returns:
    True if there exists a field that contains a user specified argument.
  """
    for value in kwargs.values():
        if isinstance(value, bool):
            return True
        elif value:
            return True
    return False