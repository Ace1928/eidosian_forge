from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
from googlecloudsdk.calliope import arg_parsers
class DeleteOption(enum.Enum):
    DESTINATION_IF_UNIQUE = 'destination-if-unique'
    SOURCE_AFTER_TRANSFER = 'source-after-transfer'