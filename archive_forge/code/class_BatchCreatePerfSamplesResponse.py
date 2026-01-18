from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BatchCreatePerfSamplesResponse(_messages.Message):
    """A BatchCreatePerfSamplesResponse object.

  Fields:
    perfSamples: A PerfSample attribute.
  """
    perfSamples = _messages.MessageField('PerfSample', 1, repeated=True)