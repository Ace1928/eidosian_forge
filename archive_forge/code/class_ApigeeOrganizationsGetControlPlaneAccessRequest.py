from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsGetControlPlaneAccessRequest(_messages.Message):
    """A ApigeeOrganizationsGetControlPlaneAccessRequest object.

  Fields:
    name: Required. Resource name of the Control Plane Access. Use the
      following structure in your request:
      `organizations/{org}/controlPlaneAccess`
  """
    name = _messages.StringField(1, required=True)