from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MarketplacesolutionsProjectsLocationsPowerInstancesCreateRequest(_messages.Message):
    """A MarketplacesolutionsProjectsLocationsPowerInstancesCreateRequest
  object.

  Fields:
    parent: Required. Parent of the resource.
    powerInstance: A PowerInstance resource to be passed as the request body.
    powerInstanceId: Required. The ID to use for the instance, which will
      become the final component of the instance name. This value should be
      4-63 characters, and valid characters are /a-z[0-9]-_/.
  """
    parent = _messages.StringField(1, required=True)
    powerInstance = _messages.MessageField('PowerInstance', 2)
    powerInstanceId = _messages.StringField(3)