from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.concepts import base
from googlecloudsdk.command_lib.concepts import dependency_managers
from googlecloudsdk.command_lib.concepts import names
import six
def FinalParse(dependencies, arg_getter):
    """Lazy parser stored under args.CONCEPT_ARGS.

  Args:
    dependencies: dependency_managers.DependencyNode, the root of the tree of
      the concept's dependencies.
    arg_getter: Callable, a function that returns the parsed args namespace.

  Raises:
      googlecloudsdk.command_lib.concepts.exceptions.Error: if parsing fails.

  Returns:
    the result of parsing the root concept.
  """
    dependency_manager = dependency_managers.DependencyManager(dependencies)
    parsed_args = arg_getter()
    return dependency_manager.ParseConcept(parsed_args)