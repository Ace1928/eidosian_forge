from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from apitools.base.py import encoding
from googlecloudsdk.api_lib.domains import operations
from googlecloudsdk.api_lib.domains import registrations
from googlecloudsdk.command_lib.domains import flags
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core import yaml
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import files
def PromptWithValidator(prompt_string, validator, error_message, message=None, default=None):
    """Prompt for user input and validate output.

  Args:
    prompt_string: Message to print in the line with prompt.
    validator: Validation function (str) -> bool.
    error_message: Message to print if provided value is not correct.
    message: Optional message to print before prompt.
    default: Optional default value.

  Returns:
    Valid user provided value or default if not None and user chose it.
  """
    if message:
        log.status.Print(message)
    while True:
        if default is not None:
            answer = console_io.PromptWithDefault(message=prompt_string, default=default)
            if not answer:
                return default
        else:
            answer = console_io.PromptResponse(prompt_string)
        if validator(answer):
            return answer
        else:
            log.status.Print(error_message)