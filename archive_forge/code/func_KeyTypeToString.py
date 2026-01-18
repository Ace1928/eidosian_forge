from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import binascii
import re
import textwrap
from apitools.base.protorpclite import messages as apitools_messages
from apitools.base.py import encoding
from googlecloudsdk.api_lib.util import apis as core_apis
from googlecloudsdk.api_lib.util import messages as messages_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions as gcloud_exceptions
from googlecloudsdk.command_lib.iam import completers
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core import yaml
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import files
import six
def KeyTypeToString(key_type):
    """Get a string version of a KeyType enum.

  Args:
    key_type: An enum of either KEY_TYPES or CREATE_KEY_TYPES.

  Returns:
    The string representation of the key_type, such that
    parseKeyType(keyTypeToString(x)) is a no-op.
  """
    if key_type == KEY_TYPES.TYPE_PKCS12_FILE or key_type == CREATE_KEY_TYPES.TYPE_PKCS12_FILE:
        return 'p12'
    elif key_type == KEY_TYPES.TYPE_GOOGLE_CREDENTIALS_FILE or key_type == CREATE_KEY_TYPES.TYPE_GOOGLE_CREDENTIALS_FILE:
        return 'json'
    else:
        return 'unspecified'