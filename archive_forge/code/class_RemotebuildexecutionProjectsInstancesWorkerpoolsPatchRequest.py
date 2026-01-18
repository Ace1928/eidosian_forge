from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RemotebuildexecutionProjectsInstancesWorkerpoolsPatchRequest(_messages.Message):
    """A RemotebuildexecutionProjectsInstancesWorkerpoolsPatchRequest object.

  Fields:
    googleDevtoolsRemotebuildexecutionAdminV1alphaUpdateWorkerPoolRequest: A
      GoogleDevtoolsRemotebuildexecutionAdminV1alphaUpdateWorkerPoolRequest
      resource to be passed as the request body.
    name: WorkerPool resource name formatted as:
      `projects/[PROJECT_ID]/instances/[INSTANCE_ID]/workerpools/[POOL_ID]`.
      name should not be populated when creating a worker pool since it is
      provided in the `poolId` field.
  """
    googleDevtoolsRemotebuildexecutionAdminV1alphaUpdateWorkerPoolRequest = _messages.MessageField('GoogleDevtoolsRemotebuildexecutionAdminV1alphaUpdateWorkerPoolRequest', 1)
    name = _messages.StringField(2, required=True)