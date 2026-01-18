from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import enum
import os
import re
import stat
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.core import log
from googlecloudsdk.core.util import platforms
import six
from six.moves import urllib
class ProviderPrefix(enum.Enum):
    """Provider prefix strings for storage URLs."""
    FILE = 'file'
    GCS = 'gs'
    HDFS = 'hdfs'
    HTTP = 'http'
    HTTPS = 'https'
    POSIX = 'posix'
    S3 = 's3'