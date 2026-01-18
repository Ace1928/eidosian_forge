from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import sys
from googlecloudsdk.calliope import walker
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.resource import resource_printer
from googlecloudsdk.core.resource import resource_projector
def ListCompletionTree(cli, branch=None, out=None):
    """Lists the static completion CLI tree as a Python module file.

  Args:
    cli: The CLI.
    branch: The path of the CLI subtree to generate.
    out: The output stream to write to, sys.stdout by default.

  Returns:
    Returns the serialized static completion CLI tree.
  """
    tree = GenerateCompletionTree(cli=cli, branch=branch)
    (out or sys.stdout).write('# -*- coding: utf-8 -*- #\n"""Cloud SDK static completion CLI tree."""\n# pylint: disable=line-too-long,bad-continuation\nSTATIC_COMPLETION_CLI_TREE = ')
    resource_printer.Print(tree, print_format='json', out=out)
    return tree