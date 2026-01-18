from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.ai import constants
def ValidateDisplayName(display_name):
    """Validates the display name."""
    if display_name is not None and (not display_name):
        raise exceptions.InvalidArgumentException('--display-name', 'Display name can not be empty.')