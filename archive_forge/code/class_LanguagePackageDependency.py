from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LanguagePackageDependency(_messages.Message):
    """Indicates a language package available between this package and the
  customer's resource artifact.

  Fields:
    package: A string attribute.
    version: A string attribute.
  """
    package = _messages.StringField(1)
    version = _messages.StringField(2)