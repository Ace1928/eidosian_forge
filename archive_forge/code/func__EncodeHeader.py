from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import platform
import re
import time
import uuid
from googlecloudsdk.core import config
from googlecloudsdk.core import log
from googlecloudsdk.core import metrics
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import platforms
import six
from six.moves import urllib
from six.moves import zip  # pylint: disable=redefined-builtin
def _EncodeHeader(header, value):
    if isinstance(header, six.text_type):
        header = header.encode('utf-8')
    if isinstance(value, six.text_type):
        value = value.encode('utf-8')
    return (header, value)