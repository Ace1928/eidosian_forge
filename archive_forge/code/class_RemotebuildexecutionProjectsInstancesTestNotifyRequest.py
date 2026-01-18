from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RemotebuildexecutionProjectsInstancesTestNotifyRequest(_messages.Message):
    """A RemotebuildexecutionProjectsInstancesTestNotifyRequest object.

  Fields:
    googleDevtoolsRemotebuildexecutionAdminV1alphaTestNotifyInstanceRequest: A
      GoogleDevtoolsRemotebuildexecutionAdminV1alphaTestNotifyInstanceRequest
      resource to be passed as the request body.
    name: Name of the instance for which to send a test notification. Format:
      `projects/[PROJECT_ID]/instances/[INSTANCE_ID]`.
  """
    googleDevtoolsRemotebuildexecutionAdminV1alphaTestNotifyInstanceRequest = _messages.MessageField('GoogleDevtoolsRemotebuildexecutionAdminV1alphaTestNotifyInstanceRequest', 1)
    name = _messages.StringField(2, required=True)