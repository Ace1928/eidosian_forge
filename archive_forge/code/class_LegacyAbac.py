from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LegacyAbac(_messages.Message):
    """Configuration for the legacy Attribute Based Access Control
  authorization mode.

  Fields:
    enabled: Whether the ABAC authorizer is enabled for this cluster. When
      enabled, identities in the system, including service accounts, nodes,
      and controllers, will have statically granted permissions beyond those
      provided by the RBAC configuration or IAM.
  """
    enabled = _messages.BooleanField(1)