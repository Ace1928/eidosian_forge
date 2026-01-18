from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import os
import re
import sys
import textwrap
from googlecloudsdk.calliope import walker
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import module_util
from googlecloudsdk.core.util import files
import six
def Dump(cli, path=None, name=DEFAULT_CLI_NAME, branch=None):
    """Dumps the CLI tree to a JSON file.

  The tree is processed by cli_tree._Serialize() to minimize the JSON file size
  and generation time.

  Args:
    cli: The CLI.
    path: The JSON file path to dump to, the standard output if '-', the default
      CLI tree path if None.
    name: The CLI name.
    branch: The path of the CLI subtree to generate.

  Returns:
    The generated CLI tree.
  """
    if path is None:
        path = CliTreeConfigPath()
    tree = _GenerateRoot(cli=cli, path=path, name=name, branch=branch)
    if path == '-':
        _DumpToFile(tree, sys.stdout)
    else:
        with files.FileWriter(path) as f:
            _DumpToFile(tree, f)
    from googlecloudsdk.core.resource import resource_projector
    return resource_projector.MakeSerializable(tree)