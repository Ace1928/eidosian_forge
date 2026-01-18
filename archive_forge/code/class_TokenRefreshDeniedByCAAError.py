from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import textwrap
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
class TokenRefreshDeniedByCAAError(TokenRefreshError):
    """Raises when token refresh is denied by context aware access policies."""

    def __init__(self, error, for_adc=False):
        from googlecloudsdk.core import context_aware
        compiled_msg = '{}\n\n{}'.format(error, context_aware.CONTEXT_AWARE_ACCESS_HELP_MSG)
        super(TokenRefreshDeniedByCAAError, self).__init__(compiled_msg, for_adc=for_adc, should_relogin=False)