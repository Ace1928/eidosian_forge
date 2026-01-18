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
def _GetRawManPageText(self):
    """Returns the raw man page text."""
    session = requests.GetSession()
    url = 'http://man7.org/linux/man-pages/man1/{}.1.html'.format(self.command_name)
    response = session.get(url)
    if response.status_code != 200:
        raise NoManPageTextForCommandError('Cannot get URL man page text for [{}].'.format(self.command_name))
    return response.text