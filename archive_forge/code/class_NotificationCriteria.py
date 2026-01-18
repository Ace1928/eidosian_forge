from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NotificationCriteria(_messages.Message):
    """Specifies when a notification should be sent.

  Fields:
    annotationChange: Specifies the annotations nested under the parent
      AssetType that should trigger notifications.
    assetChange: Specifies the asset changes that should trigger
      notifications.
    derivedAssetChange: Specifies the transformer invocations triggered by the
      derived asset rule that should publish Pub/Sub messages on event
      changes.
    transformationChange: Specifies the transformer invocations triggered by
      the transformation rule that should publish Pub/Sub messages on event
      changes.
  """
    annotationChange = _messages.MessageField('AnnotationChange', 1)
    assetChange = _messages.MessageField('AssetChange', 2)
    derivedAssetChange = _messages.MessageField('DerivedAssetChange', 3)
    transformationChange = _messages.MessageField('TransformationChange', 4)