from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Library(_messages.Message):
    """Third-party Python runtime library that is required by the application.

  Fields:
    name: Name of the library. Example: "django".
    version: Version of the library to select, or "latest".
  """
    name = _messages.StringField(1)
    version = _messages.StringField(2)