from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BatchProjectsLocationsResourceAllowancesGetRequest(_messages.Message):
    """A BatchProjectsLocationsResourceAllowancesGetRequest object.

  Fields:
    name: Required. ResourceAllowance name.
  """
    name = _messages.StringField(1, required=True)