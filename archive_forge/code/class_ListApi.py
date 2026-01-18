from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
from googlecloudsdk.calliope import arg_parsers
class ListApi(enum.Enum):
    LIST_OBJECTS = 'LIST_OBJECTS'
    LIST_OBJECTS_V2 = 'LIST_OBJECTS_V2'