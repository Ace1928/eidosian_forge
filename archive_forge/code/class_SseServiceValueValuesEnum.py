from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SseServiceValueValuesEnum(_messages.Enum):
    """Immutable. SSE service provider

    Values:
      SSE_SERVICE_UNSPECIFIED: The default value. This value is used if the
        state is omitted.
      PALO_ALTO_PRISMA_ACCESS: [Palo Alto Networks Prisma
        Access](https://www.paloaltonetworks.com/sase/access).
      SYMANTEC_CLOUD_SWG: Symantec Cloud SWG is not fully supported yet - see
        b/323856877.
    """
    SSE_SERVICE_UNSPECIFIED = 0
    PALO_ALTO_PRISMA_ACCESS = 1
    SYMANTEC_CLOUD_SWG = 2