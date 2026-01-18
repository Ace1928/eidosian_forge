from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RestoreBackupRequest(_messages.Message):
    """Request message for restoring a Backup instance.

  Fields:
    computeInstanceRestoreProperties: Compute Engine instance properties to be
      overridden during restore.
    computeInstanceTargetEnvironment: Compute Engine target environment to be
      used during restore.
    requestId: Optional. An optional request ID to identify requests. Specify
      a unique request ID so that if you must retry your request, the server
      will know to ignore the request if it has already been completed. The
      server will guarantee that for at least 60 minutes after the first
      request. For example, consider a situation where you make an initial
      request and the request times out. If you make the request again with
      the same request ID, the server can check if original operation with the
      same request ID was received, and if so, will ignore the second request.
      This prevents clients from accidentally creating duplicate commitments.
      The request ID must be a valid UUID with the exception that zero UUID is
      not supported (00000000-0000-0000-0000-000000000000).
  """
    computeInstanceRestoreProperties = _messages.MessageField('ComputeInstanceRestoreProperties', 1)
    computeInstanceTargetEnvironment = _messages.MessageField('ComputeInstanceTargetEnvironment', 2)
    requestId = _messages.StringField(3)