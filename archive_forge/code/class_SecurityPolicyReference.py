from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecurityPolicyReference(_messages.Message):
    """A SecurityPolicyReference object.

  Fields:
    securityPolicy: A string attribute.
  """
    securityPolicy = _messages.StringField(1)