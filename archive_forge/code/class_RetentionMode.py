from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
from googlecloudsdk.api_lib.storage import cloud_api
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
class RetentionMode(enum.Enum):
    """Enum class for specifying the retention mode."""
    LOCKED = 'Locked'
    UNLOCKED = 'Unlocked'