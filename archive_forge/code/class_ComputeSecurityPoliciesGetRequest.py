from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComputeSecurityPoliciesGetRequest(_messages.Message):
    """A ComputeSecurityPoliciesGetRequest object.

  Fields:
    project: Project ID for this request.
    securityPolicy: Name of the security policy to get.
  """
    project = _messages.StringField(1, required=True)
    securityPolicy = _messages.StringField(2, required=True)