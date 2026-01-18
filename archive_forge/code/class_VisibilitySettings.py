from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VisibilitySettings(_messages.Message):
    """Settings that control which features of the service are visible to the
  consumer project.

  Fields:
    visibilityLabels: The set of visibility labels that are used to determine
      what API surface is visible to calls made by this project. The visible
      surface is a union of the surface features associated with each label
      listed here, plus the publicly visible (unrestricted) surface.  The
      service producer may add or remove labels at any time. The service
      consumer may add a label if the calling user has been granted permission
      to do so by the producer.  The service consumer may also remove any
      label at any time.
  """
    visibilityLabels = _messages.StringField(1, repeated=True)