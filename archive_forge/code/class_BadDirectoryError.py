from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import hashlib
import os
from googlecloudsdk.api_lib.storage import storage_api
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core.util import files as file_utils
import six
from six.moves import zip
class BadDirectoryError(exceptions.Error):
    """Indicates that a provided directory for upload was empty."""