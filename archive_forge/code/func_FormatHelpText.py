from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import six
def FormatHelpText(field_name, required, help_text=None):
    """Defaults and formats specific attribute of help text.

  Args:
    field_name: None | str, attribute that is being set by flag
    required: bool, whether the flag is required
    help_text: None | str, text that describes the flag

  Returns:
    help text formatted as `{type} {required}, {help}`
  """
    if help_text:
        defaulted_help_text = help_text
    elif field_name:
        defaulted_help_text = 'Sets `{}` value.'.format(field_name)
    else:
        defaulted_help_text = 'Sets value.'
    if required:
        return 'Required, {}'.format(defaulted_help_text)
    else:
        return defaulted_help_text