from a man-ish style runtime document.
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import json
import os
import re
import shlex
import subprocess
import tarfile
import textwrap
from googlecloudsdk.calliope import cli_tree
from googlecloudsdk.command_lib.static_completion import generate as generate_static
from googlecloudsdk.command_lib.static_completion import lookup
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import requests
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.resource import resource_printer
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files
import six
from six.moves import range
def SubTree(self, path):
    """Generates and returns the CLI subtree rooted at path."""
    command = _Command(path)
    text = self.Run(path[1:] + ['--help'])
    collector = _KubectlCollector(text)
    while True:
        heading, content = collector.Collect()
        if not heading:
            break
        elif heading == 'COMMANDS':
            for line in content:
                try:
                    name = line.split()[0]
                except IndexError:
                    continue
                command[cli_tree.LOOKUP_IS_GROUP] = True
                command[cli_tree.LOOKUP_COMMANDS][name] = self.SubTree(path + [name])
        elif heading in ('DESCRIPTION', 'EXAMPLES'):
            command[cli_tree.LOOKUP_SECTIONS][heading] = '\n'.join(content)
        elif heading == 'FLAGS':
            self.AddFlags(command, content)
    return command