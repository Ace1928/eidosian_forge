from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TransferapplianceProjectsLocationsAppliancesGetRequest(_messages.Message):
    """A TransferapplianceProjectsLocationsAppliancesGetRequest object.

  Fields:
    name: Required. Name of the resource.
  """
    name = _messages.StringField(1, required=True)