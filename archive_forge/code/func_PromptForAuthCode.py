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
def PromptForAuthCode():
    """Prompts the user to enter the auth code."""
    message = "Please provide the authorization code from the domain's current registrar to transfer the domain."
    log.status.Print(message)
    auth_code = console_io.PromptPassword(prompt='Authorization code:  ', error_message=' Authorization code must not be empty.', validation_callable=ValidateNonEmpty)
    return auth_code