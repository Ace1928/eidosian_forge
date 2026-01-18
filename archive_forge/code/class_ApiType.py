from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
from googlecloudsdk.calliope import arg_parsers
@enum.unique
class ApiType(enum.Enum):
    """This API type is used to differentiate between the classification types of Create requests and Update requests."""
    CREATE = 'create'
    UPDATE = 'update'