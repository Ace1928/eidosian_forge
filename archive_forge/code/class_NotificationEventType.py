from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
from googlecloudsdk.command_lib.storage import storage_url
class NotificationEventType(enum.Enum):
    """Used to filter what events a notification configuration notifies on."""
    OBJECT_ARCHIVE = 'OBJECT_ARCHIVE'
    OBJECT_DELETE = 'OBJECT_DELETE'
    OBJECT_FINALIZE = 'OBJECT_FINALIZE'
    OBJECT_METADATA_UPDATE = 'OBJECT_METADATA_UPDATE'