from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GatewaySecurityPolicy(_messages.Message):
    """The GatewaySecurityPolicy resource contains a collection of
  GatewaySecurityPolicyRules and associated metadata.

  Fields:
    createTime: Output only. The timestamp when the resource was created.
    description: Optional. Free-text description of the resource.
    name: Required. Name of the resource. Name is of the form projects/{projec
      t}/locations/{location}/gatewaySecurityPolicies/{gateway_security_policy
      } gateway_security_policy should match the
      pattern:(^[a-z]([a-z0-9-]{0,61}[a-z0-9])?$).
    tlsInspectionPolicy: Optional. Name of a TLS Inspection Policy resource
      that defines how TLS inspection will be performed for any rule(s) which
      enables it.
    updateTime: Output only. The timestamp when the resource was updated.
  """
    createTime = _messages.StringField(1)
    description = _messages.StringField(2)
    name = _messages.StringField(3)
    tlsInspectionPolicy = _messages.StringField(4)
    updateTime = _messages.StringField(5)