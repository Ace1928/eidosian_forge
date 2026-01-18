from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDatacatalogV1ReconcileTagsRequest(_messages.Message):
    """Request message for ReconcileTags.

  Fields:
    forceDeleteMissing: If set to `true`, deletes entry tags related to a tag
      template not listed in the tags source from an entry. If set to `false`,
      unlisted tags are retained.
    tagTemplate: Required. The name of the tag template, which is used for
      reconciliation.
    tags: A list of tags to apply to an entry. A tag can specify a tag
      template, which must be the template specified in the
      `ReconcileTagsRequest`. The sole entry and each of its columns must be
      mentioned at most once.
  """
    forceDeleteMissing = _messages.BooleanField(1)
    tagTemplate = _messages.StringField(2)
    tags = _messages.MessageField('GoogleCloudDatacatalogV1Tag', 3, repeated=True)