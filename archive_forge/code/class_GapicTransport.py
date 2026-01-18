from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
from googlecloudsdk.core import exceptions
class GapicTransport(enum.Enum):
    """Enum options for Gapic Clients."""
    GRPC = 1
    GRPC_ASYNCIO = 2
    REST = 3