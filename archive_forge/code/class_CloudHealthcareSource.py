from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudHealthcareSource(_messages.Message):
    """Cloud Healthcare API resource.

  Fields:
    name: Full path of a Cloud Healthcare API resource.
  """
    name = _messages.StringField(1)