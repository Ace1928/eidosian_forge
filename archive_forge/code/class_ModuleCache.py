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
class ModuleCache(object):
    """A local completer module cache item to minimize intra-command latency.

  Some CLI tree positionals and flag values have completers that are specified
  by module paths. These path strings point to a completer method or class that
  can be imported at run-time. The ModuleCache keeps track of modules that have
  already been imported, the most recent completeion result, and a timeout for
  the data. This saves on import lookup, and more importantly, repeated
  completion requests within a short window. Users really love that TAB key.

  Attributes:
    _TIMEOUT: Newly updated choices stale after this many seconds.
    completer_class: The completer class.
    coshell: The coshell object.
    choices: The cached choices.
    stale: choices stale after this time.
  """
    _TIMEOUT = 60

    def __init__(self, completer_class):
        self.completer_class = completer_class
        self.choices = None
        self.stale = 0
        self.timeout = ModuleCache._TIMEOUT