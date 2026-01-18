from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TranslateProjectsLocationsGlossariesDeleteRequest(_messages.Message):
    """A TranslateProjectsLocationsGlossariesDeleteRequest object.

  Fields:
    name: Required. The name of the glossary to delete.
  """
    name = _messages.StringField(1, required=True)