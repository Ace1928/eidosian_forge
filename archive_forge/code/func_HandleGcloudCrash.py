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
def HandleGcloudCrash(err):
    """Checks if installation error occurred, then proceeds with Error Reporting.

  Args:
    err: Exception err.
  """
    err_string = console_attr.SafeText(err)
    log.file_only_logger.exception('BEGIN CRASH STACKTRACE')
    if _IsInstallationCorruption(err):
        _PrintInstallationAction(err, err_string)
    else:
        log.error('gcloud crashed ({0}): {1}'.format(getattr(err, 'error_name', type(err).__name__), err_string))
        if 'certificate verify failed' in err_string:
            log.err.Print("\ngcloud's default CA certificates failed to verify your connection, which can happen if you are behind a proxy or firewall.")
            log.err.Print('To use a custom CA certificates file, please run the following command:')
            log.err.Print('  gcloud config set core/custom_ca_certs_file /path/to/ca_certs')
        ReportError(is_crash=True)
        log.err.Print('\nIf you would like to report this issue, please run the following command:')
        log.err.Print('  gcloud feedback')
        log.err.Print('\nTo check gcloud for common problems, please run the following command:')
        log.err.Print('  gcloud info --run-diagnostics')