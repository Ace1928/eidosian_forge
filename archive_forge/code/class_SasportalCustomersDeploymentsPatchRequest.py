from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SasportalCustomersDeploymentsPatchRequest(_messages.Message):
    """A SasportalCustomersDeploymentsPatchRequest object.

  Fields:
    name: Output only. Resource name.
    sasPortalDeployment: A SasPortalDeployment resource to be passed as the
      request body.
    updateMask: Fields to be updated.
  """
    name = _messages.StringField(1, required=True)
    sasPortalDeployment = _messages.MessageField('SasPortalDeployment', 2)
    updateMask = _messages.StringField(3)