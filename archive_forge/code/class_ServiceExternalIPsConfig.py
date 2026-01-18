from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ServiceExternalIPsConfig(_messages.Message):
    """Config to block services with externalIPs field.

  Fields:
    enabled: Whether Services with ExternalIPs field are allowed or not.
  """
    enabled = _messages.BooleanField(1)