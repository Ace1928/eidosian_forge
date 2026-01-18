from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AnnotationSource(_messages.Message):
    """AnnotationSource holds the source information of the annotation.

  Fields:
    cloudHealthcareSource: Cloud Healthcare API resource.
  """
    cloudHealthcareSource = _messages.MessageField('CloudHealthcareSource', 1)