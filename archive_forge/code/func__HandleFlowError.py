from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import json
import textwrap
from googlecloudsdk.command_lib.util import check_browser
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import files
import six
def _HandleFlowError(exc, default_help_msg):
    """Prints help messages when auth flow throws errors."""
    from googlecloudsdk.core import context_aware
    if context_aware.IsContextAwareAccessDeniedError(exc):
        log.error(context_aware.CONTEXT_AWARE_ACCESS_HELP_MSG)
    else:
        log.error(default_help_msg)