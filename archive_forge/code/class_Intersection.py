from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Intersection(_messages.Message):
    """A GcRule which deletes cells matching all of the given rules.

  Fields:
    rules: Only delete cells which would be deleted by every element of
      `rules`.
  """
    rules = _messages.MessageField('GcRule', 1, repeated=True)