from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import binascii
import copy
import io
import json
import textwrap
from apitools.base.py import encoding
from apitools.base.py import exceptions as api_exceptions
from googlecloudsdk.api_lib.bigtable import util
from googlecloudsdk.api_lib.util import exceptions
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.core import yaml
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.resource import resource_diff
from googlecloudsdk.core.util import edit
import six
def BuildUpdateAuthorizedViewFileContents(current_authorized_view, pre_encoded):
    """Builds the help text for updating an existing authorized view.

  Args:
    current_authorized_view: The current authorized view resource object.
    pre_encoded: When pre_encoded is False, user is passing utf-8 values for
      binary fields in the authorized view definition and expecting us to apply
      base64 encoding. Thus, we should also display utf-8 values in the help
      text, which requires base64 decoding the binary fields in the
      `current_authorized_view`.

  Returns:
    A string containing the help text for update authorized view.
  """
    buf = io.StringIO()
    for line in UPDATE_HELP.splitlines():
        buf.write('#')
        if line:
            buf.write(' ')
        buf.write(line)
        buf.write('\n')
    serialized_original_authorized_view = SerializeToJsonOrYaml(current_authorized_view, pre_encoded)
    for line in serialized_original_authorized_view.splitlines():
        buf.write('#')
        if line:
            buf.write(' ')
        buf.write(line)
        buf.write('\n')
    buf.write('\n')
    return buf.getvalue()