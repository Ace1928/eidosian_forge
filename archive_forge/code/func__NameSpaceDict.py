from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import io
import os
import sys
import threading
import time
from googlecloudsdk.calliope import parser_completer
from googlecloudsdk.command_lib.interactive import parser
from googlecloudsdk.command_lib.meta import generate_cli_trees
from googlecloudsdk.core import module_util
from googlecloudsdk.core.console import console_attr
from prompt_toolkit import completion
import six
def _NameSpaceDict(args):
    """Returns a namespace dict given parsed CLI tree args."""
    namespace = {}
    name = None
    for arg in args:
        if arg.token_type == parser.ArgTokenType.POSITIONAL:
            name = arg.tree.get(parser.LOOKUP_NAME)
            value = arg.value
        elif arg.token_type == parser.ArgTokenType.FLAG:
            name = arg.tree.get(parser.LOOKUP_NAME)
            if name:
                if name.startswith('--'):
                    name = name[2:]
                name = name.replace('-', '_')
            continue
        elif not name:
            continue
        elif arg.token_type == parser.ArgTokenType.FLAG_ARG:
            value = arg.value
        else:
            continue
        namespace[name] = value
    return namespace