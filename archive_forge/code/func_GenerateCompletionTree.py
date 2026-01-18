from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import sys
from googlecloudsdk.calliope import walker
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.resource import resource_printer
from googlecloudsdk.core.resource import resource_projector
def GenerateCompletionTree(cli, branch=None, ignore_load_errors=False):
    """Generates and returns the static completion CLI tree.

  Args:
    cli: The CLI.
    branch: The path of the CLI subtree to generate.
    ignore_load_errors: Ignore CLI tree load errors if True.

  Returns:
    Returns the serialized static completion CLI tree.
  """
    with progress_tracker.ProgressTracker('Generating the static completion CLI tree.'):
        return resource_projector.MakeSerializable(_CompletionTreeGenerator(cli, branch=branch, ignore_load_errors=ignore_load_errors).Walk())