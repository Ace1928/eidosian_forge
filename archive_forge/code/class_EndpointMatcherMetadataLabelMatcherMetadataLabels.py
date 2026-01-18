from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EndpointMatcherMetadataLabelMatcherMetadataLabels(_messages.Message):
    """Defines a name-pair value for a single label.

  Fields:
    labelName: Required. Label name presented as key in xDS Node Metadata.
    labelValue: Required. Label value presented as value corresponding to the
      above key, in xDS Node Metadata.
  """
    labelName = _messages.StringField(1)
    labelValue = _messages.StringField(2)