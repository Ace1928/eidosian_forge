from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import times
import six
def GetRedactColorFromString(color_string):
    """Convert color_string into GooglePrivacyDlpV2Color.

  Creates a GooglePrivacyDlpV2Color message from input string to use for image
  redaction.

  Args:
    color_string: str, string representing red, green and blue color saturation
      percentages as float values between 0.0 and 1.0. For example, `black =
      0,0,0`, `red = 1.0,0,0`, `white = 1.0,1.0,1.0` etc.

  Returns:
    GooglePrivacyDlpV2Color, color message.

  Raises:
    RedactColorError if color_string is improperly formatted.
  """
    color_msg = _GetMessageClass('GooglePrivacyDlpV2Color')
    red, green, blue = _ValidateAndParseColors(color_string)
    return color_msg(red=red, blue=blue, green=green)