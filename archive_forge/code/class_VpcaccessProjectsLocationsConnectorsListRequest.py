from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VpcaccessProjectsLocationsConnectorsListRequest(_messages.Message):
    """A VpcaccessProjectsLocationsConnectorsListRequest object.

  Fields:
    pageSize: Maximum number of functions to return per call.
    pageToken: Continuation token.
    parent: Required. The project and location from which the routes should be
      listed.
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)