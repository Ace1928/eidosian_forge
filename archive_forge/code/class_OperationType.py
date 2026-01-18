from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
from googlecloudsdk.api_lib.workbench import util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
class OperationType(enum.Enum):
    CREATE = (log.CreatedResource, 'created')
    UPDATE = (log.UpdatedResource, 'updated')
    UPGRADE = (log.UpdatedResource, 'upgraded')
    ROLLBACK = (log.UpdatedResource, 'rolled back')
    DELETE = (log.DeletedResource, 'deleted')
    RESET = (log.ResetResource, 'reset')