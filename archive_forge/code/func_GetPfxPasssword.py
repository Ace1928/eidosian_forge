from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.active_directory import exceptions
from googlecloudsdk.command_lib.active_directory import util
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core.console import console_io
def GetPfxPasssword():
    """Prompt for user input of pfx password."""
    cred = console_io.PromptPassword('Please enter the password used the encrypt the pfx certificate: ')
    return cred