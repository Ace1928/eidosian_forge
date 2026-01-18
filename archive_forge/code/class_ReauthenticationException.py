from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import textwrap
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
class ReauthenticationException(Error):
    """Exceptions that tells the user to retry his command or run auth login."""

    def __init__(self, message, for_adc=False):
        login_command = ADC_LOGIN_COMMAND if for_adc else AUTH_LOGIN_COMMAND
        super(ReauthenticationException, self).__init__(textwrap.dedent('        {message}\n        Please retry your command or run:\n\n          $ {login_command}\n\n        to obtain new credentials.'.format(message=message, login_command=login_command)))