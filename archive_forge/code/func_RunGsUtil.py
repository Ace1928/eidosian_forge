from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import contextlib
import datetime
import locale
import logging
import os
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
import boto
from boto import config
from boto.exception import StorageResponseError
from boto.s3.deletemarker import DeleteMarker
from boto.storage_uri import BucketStorageUri
import gslib
from gslib.boto_translation import BotoTranslation
from gslib.cloud_api import PreconditionException
from gslib.cloud_api import Preconditions
from gslib.discard_messages_queue import DiscardMessagesQueue
from gslib.exception import CommandException
from gslib.gcs_json_api import GcsJsonApi
from gslib.kms_api import KmsApi
from gslib.project_id import GOOG_PROJ_ID_HDR
from gslib.project_id import PopulateProjectId
from gslib.tests.testcase import base
import gslib.tests.util as util
from gslib.tests.util import InvokedFromParFile
from gslib.tests.util import ObjectToURI as suri
from gslib.tests.util import RUN_S3_TESTS
from gslib.tests.util import SetBotoConfigForTest
from gslib.tests.util import SetEnvironmentForTest
from gslib.tests.util import unittest
from gslib.tests.util import USING_JSON_API
import gslib.third_party.storage_apitools.storage_v1_messages as apitools_messages
from gslib.utils.constants import UTF8
from gslib.utils.encryption_helper import Base64Sha256FromBase64EncryptionKey
from gslib.utils.encryption_helper import CryptoKeyWrapperFromKey
from gslib.utils.hashing_helper import Base64ToHexHash
from gslib.utils.metadata_util import CreateCustomMetadata
from gslib.utils.metadata_util import GetValueFromObjectCustomMetadata
from gslib.utils.posix_util import ATIME_ATTR
from gslib.utils.posix_util import GID_ATTR
from gslib.utils.posix_util import MODE_ATTR
from gslib.utils.posix_util import MTIME_ATTR
from gslib.utils.posix_util import UID_ATTR
from gslib.utils.retry_util import Retry
import six
from six.moves import range
def RunGsUtil(self, cmd, return_status=False, return_stdout=False, return_stderr=False, expected_status=0, stdin=None, env_vars=None, force_gsutil=False):
    """Runs the gsutil command.

    Args:
      cmd: The command to run, as a list, e.g. ['cp', 'foo', 'bar']
      return_status: If True, the exit status code is returned.
      return_stdout: If True, the standard output of the command is returned.
      return_stderr: If True, the standard error of the command is returned.
      expected_status: The expected return code. If not specified, defaults to
                       0. If the return code is a different value, an exception
                       is raised.
      stdin: A string of data to pipe to the process as standard input.
      env_vars: A dictionary of variables to extend the subprocess's os.environ
                with.
      force_gsutil: If True, will always run the command using gsutil,
        irrespective of the value provided for use_gcloud_storage.

    Returns:
      If multiple return_* values were specified, this method returns a tuple
      containing the desired return values specified by the return_* arguments
      (in the order those parameters are specified in the method definition).
      If only one return_* value was specified, that value is returned directly
      rather than being returned within a 1-tuple.
    """
    full_gsutil_command = util.GetGsutilCommand(cmd, force_gsutil=force_gsutil)
    process = util.GetGsutilSubprocess(full_gsutil_command, env_vars=env_vars)
    c_out = util.CommunicateWithTimeout(process, stdin=stdin)
    stdout = c_out[0].replace(os.linesep, '\n')
    stderr = c_out[1].replace(os.linesep, '\n')
    status = process.returncode
    if expected_status is not None:
        cmd = map(six.ensure_text, cmd)
        self.assertEqual(int(status), int(expected_status), msg='Expected status {}, got {}.\nCommand:\n{}\n\nstderr:\n{}'.format(expected_status, status, ' '.join(cmd), stderr))
    toreturn = []
    if return_status:
        toreturn.append(status)
    if return_stdout:
        toreturn.append(stdout)
    if return_stderr:
        toreturn.append(stderr)
    if len(toreturn) == 1:
        return toreturn[0]
    elif toreturn:
        return tuple(toreturn)