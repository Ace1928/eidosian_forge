from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
from googlecloudsdk.command_lib.storage import storage_url
class FieldsScope(enum.Enum):
    """Values used to determine fields and projection values for API calls."""
    FULL = 1
    NO_ACL = 2
    RSYNC = 3
    SHORT = 4