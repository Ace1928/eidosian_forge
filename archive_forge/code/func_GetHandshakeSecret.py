from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core.console import console_io
def GetHandshakeSecret():
    """Prompt for user input of handshake secret with target domain."""
    unused_cred = console_io.PromptPassword('Please enter handshake secret with target domain. The secret will not be stored: ')
    return unused_cred