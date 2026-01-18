from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class InfrastructureValueValuesEnum(_messages.Enum):
    """Output only. The type of underlying resources used to create the
    connection.

    Values:
      INFRASTRUCTURE_UNSPECIFIED: An invalid infrastructure as the default
        case.
      PSC: Private Service Connect is used for connections.
    """
    INFRASTRUCTURE_UNSPECIFIED = 0
    PSC = 1