from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleDevtoolsRemotebuildexecutionAdminV1alphaListInstancesResponse(_messages.Message):
    """The response used for `ListInstances`.

  Fields:
    instances: The list of instances in a given project.
    unreachable: Unreachable regions.
  """
    instances = _messages.MessageField('GoogleDevtoolsRemotebuildexecutionAdminV1alphaInstance', 1, repeated=True)
    unreachable = _messages.StringField(2, repeated=True)