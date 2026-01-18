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
class CacheArg(object):
    """A completion cache arg."""

    def __init__(self, prefix, completions):
        self.prefix = prefix
        self.completions = completions
        self.dirs = {}

    def IsValid(self):
        return self.completions is not None

    def Invalidate(self):
        self.command_count = _INVALID_ARG_COMMAND_COUNT
        self.completions = None
        self.dirs = {}