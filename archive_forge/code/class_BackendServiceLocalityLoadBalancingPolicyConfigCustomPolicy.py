from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BackendServiceLocalityLoadBalancingPolicyConfigCustomPolicy(_messages.Message):
    """The configuration for a custom policy implemented by the user and
  deployed with the client.

  Fields:
    data: An optional, arbitrary JSON object with configuration data,
      understood by a locally installed custom policy implementation.
    name: Identifies the custom policy. The value should match the name of a
      custom implementation registered on the gRPC clients. It should follow
      protocol buffer message naming conventions and include the full path
      (for example, myorg.CustomLbPolicy). The maximum length is 256
      characters. Do not specify the same custom policy more than once for a
      backend. If you do, the configuration is rejected. For an example of how
      to use this field, see Use a custom policy.
  """
    data = _messages.StringField(1)
    name = _messages.StringField(2)