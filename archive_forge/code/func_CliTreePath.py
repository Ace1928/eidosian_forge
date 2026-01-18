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
def CliTreePath(name=DEFAULT_CLI_NAME, directory=None):
    """Returns the CLI tree file path for name, default if directory is None."""
    return os.path.join(directory or CliTreeDir(), name + '.json')