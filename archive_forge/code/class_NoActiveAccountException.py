from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import textwrap
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
class NoActiveAccountException(AuthenticationException):
    """Exception for when there are no valid active credentials."""

    def __init__(self, active_config_path=None):
        if active_config_path:
            if not os.path.exists(active_config_path):
                log.warning('Could not open the configuration file: [%s].', active_config_path)
        super(NoActiveAccountException, self).__init__('You do not currently have an active account selected.')