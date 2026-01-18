from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworkservicesProjectsLocationsGrpcRoutesGetRequest(_messages.Message):
    """A NetworkservicesProjectsLocationsGrpcRoutesGetRequest object.

  Fields:
    name: Required. A name of the GrpcRoute to get. Must be in the format
      `projects/*/locations/global/grpcRoutes/*`.
  """
    name = _messages.StringField(1, required=True)