from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1RevisionStatus(_messages.Message):
    """The status of a specific resource revision.

  Fields:
    errors: Errors reported when attempting to load this revision.
    jsonSpec: The json content of the resource revision. Large specs should be
      sent individually via the spec field to avoid hitting request size
      limits.
    replicas: The number of replicas that have successfully loaded this
      revision.
    revisionId: The revision of the resource.
  """
    errors = _messages.MessageField('GoogleCloudApigeeV1UpdateError', 1, repeated=True)
    jsonSpec = _messages.StringField(2)
    replicas = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    revisionId = _messages.StringField(4)