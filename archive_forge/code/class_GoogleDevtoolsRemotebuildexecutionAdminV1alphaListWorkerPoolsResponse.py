from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleDevtoolsRemotebuildexecutionAdminV1alphaListWorkerPoolsResponse(_messages.Message):
    """A GoogleDevtoolsRemotebuildexecutionAdminV1alphaListWorkerPoolsResponse
  object.

  Fields:
    workerPools: The list of worker pools in a given instance.
  """
    workerPools = _messages.MessageField('GoogleDevtoolsRemotebuildexecutionAdminV1alphaWorkerPool', 1, repeated=True)