from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import os
import subprocess
import sys
import webbrowser
from googlecloudsdk.command_lib.interactive import parser
from googlecloudsdk.core import log
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files
import six
def _GetReferenceURL(cli, line, pos=None, man_page=False):
    """Determine the reference url of the command/group preceding the pos.

  Args:
    cli: the prompt CLI object
    line: a string with the current string directly from the shell.
    pos: the position of the cursor on the line.
    man_page: Return help/man page command line if True.

  Returns:
    A string containing the URL of the reference page.
  """
    mappers = {'bq': BqReferenceMapper, 'gcloud': GcloudReferenceMapper, 'gsutil': GsutilReferenceMapper, 'kubectl': KubectlReferenceMapper}
    if pos is None:
        pos = len(line)
    args = []
    for arg in cli.parser.ParseCommand(line):
        if arg.start < pos and (not args or arg.tree.get(parser.LOOKUP_COMMANDS) or arg.token_type in (parser.ArgTokenType.COMMAND, parser.ArgTokenType.GROUP)):
            args.append(arg.value)
    if not args:
        if line:
            return None
        args = ['gcloud', 'alpha', 'interactive']
    mapper_class = mappers.get(args[0], UnknownReferenceMapper)
    mapper = mapper_class(cli, args)
    return mapper.GetMan() if man_page else mapper.GetURL()