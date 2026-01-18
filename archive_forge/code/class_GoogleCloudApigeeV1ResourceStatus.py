from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1ResourceStatus(_messages.Message):
    """The status of a resource loaded in the runtime.

  Fields:
    resource: The resource name. Currently only two resources are supported:
      EnvironmentGroup - organizations/{org}/envgroups/{envgroup}
      EnvironmentConfig -
      organizations/{org}/environments/{environment}/deployedConfig
    revisions: Revisions of the resource currently deployed in the instance.
    totalReplicas: The total number of replicas that should have this
      resource.
    uid: The uid of the resource. In the unexpected case that the instance has
      multiple uids for the same name, they should be reported under separate
      ResourceStatuses.
  """
    resource = _messages.StringField(1)
    revisions = _messages.MessageField('GoogleCloudApigeeV1RevisionStatus', 2, repeated=True)
    totalReplicas = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    uid = _messages.StringField(4)