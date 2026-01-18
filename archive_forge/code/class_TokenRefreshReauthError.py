from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import textwrap
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
class TokenRefreshReauthError(ReauthenticationException):
    """An exception raised when the auth tokens fail to refresh due to reauth."""

    def __init__(self, error, for_adc=False):
        message = 'There was a problem reauthenticating while refreshing your current auth tokens: {0}'.format(error)
        super(TokenRefreshReauthError, self).__init__(message, for_adc=for_adc)