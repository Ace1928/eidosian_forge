from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CaSourceValueValuesEnum(_messages.Enum):
    """Optional. Certificate Authority (CA) source. Only CA_SOURCE_MANAGED is
    supported currently, and is the default value.

    Values:
      CA_SOURCE_UNSPECIFIED: Certificate Authority (CA) source not specified.
        Defaults to CA_SOURCE_MANAGED.
      CA_SOURCE_MANAGED: Certificate Authority (CA) managed by the AlloyDB
        Cluster.
    """
    CA_SOURCE_UNSPECIFIED = 0
    CA_SOURCE_MANAGED = 1