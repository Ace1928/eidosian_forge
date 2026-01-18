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
def FindTreeFile(self, directories):
    """Returns (path,f) open for read for the first CLI tree in directories."""
    for directory in directories or _GetDirectories(warn_on_exceptions=True):
        path = os.path.join(directory or '.', self.command_name) + '.json'
        try:
            return (path, files.FileReader(path))
        except files.Error:
            pass
    return (path, None)