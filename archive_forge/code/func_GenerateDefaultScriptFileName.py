from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import re
from googlecloudsdk.calliope.exceptions import core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import platforms
from googlecloudsdk.core.util import times
from mako import runtime
from mako import template
def GenerateDefaultScriptFileName():
    """Generate a default filename for import script."""
    suffix = 'cmd' if platforms.OperatingSystem.IsWindows() else 'sh'
    return IMPORT_SCRIPT_DEFAULT_NAME.format(ts=times.FormatDateTime(times.Now(), IMPORT_DATE_FORMAT), suffix=suffix)