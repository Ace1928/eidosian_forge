from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LocalObjectReference(_messages.Message):
    """Not supported by Cloud Run. LocalObjectReference contains enough
  information to let you locate the referenced object inside the same
  namespace.

  Fields:
    name: Name of the referent.
  """
    name = _messages.StringField(1)