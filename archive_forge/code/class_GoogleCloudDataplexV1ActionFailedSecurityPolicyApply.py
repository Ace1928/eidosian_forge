from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDataplexV1ActionFailedSecurityPolicyApply(_messages.Message):
    """Failed to apply security policy to the managed resource(s) under a lake,
  zone or an asset. For a lake or zone resource, one or more underlying assets
  has a failure applying security policy to the associated managed resource.

  Fields:
    asset: Resource name of one of the assets with failing security policy
      application. Populated for a lake or zone resource only.
  """
    asset = _messages.StringField(1)