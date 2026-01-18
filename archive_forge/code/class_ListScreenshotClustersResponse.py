from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListScreenshotClustersResponse(_messages.Message):
    """A ListScreenshotClustersResponse object.

  Fields:
    clusters: The set of clusters associated with an execution Always set
  """
    clusters = _messages.MessageField('ScreenshotCluster', 1, repeated=True)