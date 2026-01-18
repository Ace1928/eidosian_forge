from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import functools
import sys
import traceback
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.error_reporting import util
from googlecloudsdk.api_lib.util import apis as core_apis
from googlecloudsdk.calliope import command_loading
from googlecloudsdk.command_lib import error_reporting_util
from googlecloudsdk.core import config
from googlecloudsdk.core import log
from googlecloudsdk.core import metrics
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.util import platforms
def _IsInstallationCorruption(err):
    """Determines if the error may be from installation corruption.

  Args:
    err: Exception err.

  Returns:
    bool, True if installation error, False otherwise
  """
    return isinstance(err, command_loading.CommandLoadFailure) and isinstance(err.root_exception, ImportError)