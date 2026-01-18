from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ScanData(_messages.Message):
    """ScanData contains Cloud Key Visualizer scan data used by the caller to
  construct a visualization.

  Fields:
    data: Cloud Key Visualizer scan data. The range of time this information
      covers is captured via the above time range fields. Note, this field is
      not available to the ListScans method.
    endTime: The upper bound for when the contained data is defined.
    startTime: A range of time (inclusive) for when the contained data is
      defined. The lower bound for when the contained data is defined.
  """
    data = _messages.MessageField('VisualizationData', 1)
    endTime = _messages.StringField(2)
    startTime = _messages.StringField(3)