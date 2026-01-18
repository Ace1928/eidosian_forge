from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EdgeConfigstoreBundleBadBundleViolation(_messages.Message):
    """A message type used to describe a single bundle validation error.

  Fields:
    description: A description of why the bundle is invalid and how to fix it.
    filename: The filename (including relative path from the bundle root) in
      which the error occurred.
  """
    description = _messages.StringField(1)
    filename = _messages.StringField(2)