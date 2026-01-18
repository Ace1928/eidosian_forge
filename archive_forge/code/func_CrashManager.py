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
def CrashManager(target_function):
    """Context manager for handling gcloud crashes.

  Good for wrapping multiprocessing and multithreading target functions.

  Args:
    target_function (function): Unit test to decorate.

  Returns:
    Decorator function.
  """

    @functools.wraps(target_function)
    def Wrapper(*args, **kwargs):
        try:
            target_function(*args, **kwargs)
        except Exception as e:
            HandleGcloudCrash(e)
            if properties.VALUES.core.print_unhandled_tracebacks.GetBool():
                raise
            else:
                sys.exit(1)
    return Wrapper