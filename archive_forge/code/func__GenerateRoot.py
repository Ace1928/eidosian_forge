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
def _GenerateRoot(cli, path=None, name=DEFAULT_CLI_NAME, branch=None):
    """Generates and returns the CLI root for name."""
    from googlecloudsdk.core.console import progress_tracker
    if path == '-':
        message = 'Generating the {} CLI'.format(name)
    elif path:
        message = 'Generating the {} CLI and caching in [{}]'.format(name, path)
    else:
        message = 'Generating the {} CLI for one-time use (no SDK root)'.format(name)
    with progress_tracker.ProgressTracker(message):
        tree = CliTreeGenerator(cli, branch=branch).Walk(hidden=True)
        setattr(tree, LOOKUP_VERSION, VERSION)
        setattr(tree, LOOKUP_CLI_VERSION, _GetDefaultCliCommandVersion())
        return tree