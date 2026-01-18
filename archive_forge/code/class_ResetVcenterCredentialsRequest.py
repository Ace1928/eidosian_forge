from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ResetVcenterCredentialsRequest(_messages.Message):
    """Request message for VmwareEngine.ResetVcenterCredentials

  Fields:
    requestId: Optional. A request ID to identify requests. Specify a unique
      request ID so that if you must retry your request, the server will know
      to ignore the request if it has already been completed. The server
      guarantees that a request doesn't result in creation of duplicate
      commitments for at least 60 minutes. For example, consider a situation
      where you make an initial request and the request times out. If you make
      the request again with the same request ID, the server can check if
      original operation with the same request ID was received, and if so,
      will ignore the second request. This prevents clients from accidentally
      creating duplicate commitments. The request ID must be a valid UUID with
      the exception that zero UUID is not supported
      (00000000-0000-0000-0000-000000000000).
    username: Optional. The username of the user to be to reset the
      credentials. The default value of this field is CloudOwner@gve.local.
      The provided value should be one of the following: solution-
      user-01@gve.local, solution-user-02@gve.local, solution-
      user-03@gve.local, solution-user-04@gve.local, solution-
      user-05@gve.local, zertoadmin@gve.local.
  """
    requestId = _messages.StringField(1)
    username = _messages.StringField(2)