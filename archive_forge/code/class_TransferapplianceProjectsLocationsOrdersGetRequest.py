from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TransferapplianceProjectsLocationsOrdersGetRequest(_messages.Message):
    """A TransferapplianceProjectsLocationsOrdersGetRequest object.

  Fields:
    name: Required. Name of the resource.
  """
    name = _messages.StringField(1, required=True)