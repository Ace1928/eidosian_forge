from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import io
import mimetypes
import os
from apitools.base.py import exceptions as api_exceptions
from apitools.base.py import list_pager
from apitools.base.py import transfer
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.api_lib.util import exceptions as http_exc
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.iam import iam_util
from googlecloudsdk.core import exceptions as core_exc
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.credentials import transports
from googlecloudsdk.core.util import scaled_integer
import six
def _GetFileSize(local_path):
    try:
        return os.path.getsize(six.ensure_str(local_path))
    except os.error:
        raise exceptions.BadFileException('[{0}] not found or not accessible'.format(local_path))