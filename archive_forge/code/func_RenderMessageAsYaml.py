from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import binascii
import io
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.resource import resource_printer
from googlecloudsdk.core.util import http_encoding
def RenderMessageAsYaml(message):
    """Returns a ready-to-print string representation for the provided message.

  Args:
    message: message object

  Returns:
    A ready-to-print string representation of the message.
  """
    output_message = io.StringIO()
    resource_printer.Print(message, 'yaml', out=output_message)
    return output_message.getvalue()