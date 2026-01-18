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
def PromptForEnum(enum_mapper, enum_type, current_value):
    """Prompts the user for the new enum_type value.

  Args:
    enum_mapper: Instance of the EnumMapper.
    enum_type: A string with enum type name to print.
    current_value: Current value of the enum.

  Returns:
    The new enum choice or None if the enum shouldn't be updated.
  """
    options = list(enum_mapper.choices)
    update = console_io.PromptContinue(f'Your current {enum_type} is: {current_value}.', 'Do you want to change it', default=False)
    if not update:
        return None
    current_choice = 0
    for i, enum in enumerate(options):
        if enum == enum_mapper.GetChoiceForEnum(current_value):
            current_choice = i
    index = console_io.PromptChoice(options=options, default=current_choice, message=f'Specify new {enum_type}')
    return options[index]