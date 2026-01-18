from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1ResourceSpec(_messages.Message):
    """A resource spec, to be referenced in a ResourceStatus.

  Fields:
    json: The json content of the resource revision.
    resource: The resource name. Currently only two resources are supported:
      EnvironmentGroup - organizations/{org}/envgroups/{envgroup}
      EnvironmentConfig -
      organizations/{org}/environments/{environment}/deployedConfig
    revisionId: The revision of the resource.
    uid: The uid of the resource. In the unexpected case that the instance has
      multiple uids for the same name, they should be reported under separate
      ResourceStatuses.
  """
    json = _messages.StringField(1)
    resource = _messages.StringField(2)
    revisionId = _messages.StringField(3)
    uid = _messages.StringField(4)