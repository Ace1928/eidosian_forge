from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListMetricDescriptorsResponse(_messages.Message):
    """The ListMetricDescriptors response.

  Fields:
    metricDescriptors: The metric descriptors that are available to the
      project and that match the value of filter, if present.
    nextPageToken: If there are more results than have been returned, then
      this field is set to a non-empty value. To see the additional results,
      use that value as page_token in the next call to this method.
  """
    metricDescriptors = _messages.MessageField('MetricDescriptor', 1, repeated=True)
    nextPageToken = _messages.StringField(2)