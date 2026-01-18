from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudMlV1GetConfigResponse(_messages.Message):
    """Returns service account information associated with a project.

  Fields:
    config: A GoogleCloudMlV1Config attribute.
    serviceAccount: The service account Cloud ML uses to access resources in
      the project.
    serviceAccountProject: The project number for `service_account`.
  """
    config = _messages.MessageField('GoogleCloudMlV1Config', 1)
    serviceAccount = _messages.StringField(2)
    serviceAccountProject = _messages.IntegerField(3)