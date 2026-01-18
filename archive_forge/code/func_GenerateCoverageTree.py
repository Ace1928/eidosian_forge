from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import walker
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.resource import resource_printer
from googlecloudsdk.core.resource import resource_projector
def GenerateCoverageTree(cli, branch=None, restrict=None):
    """Generates and returns the static completion CLI tree.

  Args:
    cli: The CLI.
    branch: The path of the CLI subtree to generate.
    restrict: The paths in the tree that we are allowing the tree to walk under.

  Returns:
    Returns the serialized static completion CLI tree.
  """
    with progress_tracker.ProgressTracker('Generating the flag coverage CLI tree.'):
        return resource_projector.MakeSerializable(CoverageTreeGenerator(cli, branch, restrict=restrict).Walk())