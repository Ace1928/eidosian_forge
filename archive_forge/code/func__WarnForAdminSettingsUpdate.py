from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core.console import console_io
def _WarnForAdminSettingsUpdate():
    """Adds prompt that warns about allowed email domains update."""
    message = 'Change to instance allowed email domain requested. '
    message += 'Updating the allowed email domains from cli means the value provided will be considered as the entire list and not an amendment to the existing list of allowed email domains.'
    console_io.PromptContinue(message=message, prompt_string='Do you want to proceed with update?', cancel_on_no=True)