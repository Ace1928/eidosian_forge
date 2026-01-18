from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VirtualRepositoryConfig(_messages.Message):
    """Virtual repository configuration.

  Fields:
    upstreamPolicies: Policies that configure the upstream artifacts
      distributed by the Virtual Repository. Upstream policies cannot be set
      on a standard repository.
  """
    upstreamPolicies = _messages.MessageField('UpstreamPolicy', 1, repeated=True)