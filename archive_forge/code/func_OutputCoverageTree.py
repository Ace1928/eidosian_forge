from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import walker
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.resource import resource_printer
from googlecloudsdk.core.resource import resource_projector
def OutputCoverageTree(cli, branch=None, out=None, restrict=None):
    """Lists the flag coverage CLI tree as a Python module file.

  Args:
    cli: The CLI.
    branch: The path of the CLI subtree to generate.
    out: The output stream to write to, sys.stdout by default.
    restrict: The paths in the tree that we are allowing the tree to walk under.

  Returns:
    Returns the serialized coverage CLI tree.
  """
    tree = GenerateCoverageTree(cli=cli, branch=branch, restrict=restrict)
    resource_printer.Print(tree, print_format='json', out=out)
    return tree