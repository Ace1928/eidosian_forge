from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PathOverride(_messages.Message):
    """PathOverride. Path message defines path override for HTTP targets.

  Fields:
    path: The URI path (e.g., /users/1234). Default is an empty string.
  """
    path = _messages.StringField(1)