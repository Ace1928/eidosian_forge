from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MarketplacesolutionsProjectsLocationsPowerSshKeysGetRequest(_messages.Message):
    """A MarketplacesolutionsProjectsLocationsPowerSshKeysGetRequest object.

  Fields:
    name: Required. Name of the resource.
  """
    name = _messages.StringField(1, required=True)