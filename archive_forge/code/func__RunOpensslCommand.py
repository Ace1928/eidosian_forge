from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import hashlib
import json
import os
import subprocess
import tempfile
import time
import uuid
from apitools.base.py import encoding
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.dataproc import exceptions
from googlecloudsdk.api_lib.dataproc import storage_helpers
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.export import util as export_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import requests
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.credentials import creds as c_creds
from googlecloudsdk.core.credentials import store as c_store
from googlecloudsdk.core.util import retry
import six
def _RunOpensslCommand(self, openssl_executable, args, stdin=None):
    """Run the specified command, capturing and returning output as appropriate.

    Args:
      openssl_executable: The path to the openssl executable.
      args: The arguments to the openssl command to run.
      stdin: The input to the command.

    Returns:
      The output of the command.

    Raises:
      PersonalAuthError: If the call to openssl fails
    """
    command = [openssl_executable]
    command.extend(args)
    stderr = None
    try:
        if getattr(subprocess, 'run', None):
            proc = subprocess.run(command, input=stdin, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
            stderr = proc.stderr.decode('utf-8').strip()
            proc.check_returncode()
            return proc.stdout
        else:
            p = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, _ = p.communicate(input=stdin)
            return stdout
    except Exception as ex:
        if stderr:
            log.error('OpenSSL command "%s" failed with error message "%s"', ' '.join(command), stderr)
        raise exceptions.PersonalAuthError('Failure running openssl command: "' + ' '.join(command) + '": ' + six.text_type(ex))