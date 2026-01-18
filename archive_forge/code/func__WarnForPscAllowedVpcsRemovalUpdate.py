from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core.console import console_io
def _WarnForPscAllowedVpcsRemovalUpdate():
    """Adds prompt that warns about allowed vpcs removal."""
    message = 'Removal of instance PSC allowed vpcs requested. '
    console_io.PromptContinue(message=message, prompt_string='Do you want to proceed with removal of PSC allowed vpcs?', cancel_on_no=True)