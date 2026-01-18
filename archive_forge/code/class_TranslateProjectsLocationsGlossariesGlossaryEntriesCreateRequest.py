from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TranslateProjectsLocationsGlossariesGlossaryEntriesCreateRequest(_messages.Message):
    """A TranslateProjectsLocationsGlossariesGlossaryEntriesCreateRequest
  object.

  Fields:
    glossaryEntry: A GlossaryEntry resource to be passed as the request body.
    parent: Required. The resource name of the glossary to create the entry
      under.
  """
    glossaryEntry = _messages.MessageField('GlossaryEntry', 1)
    parent = _messages.StringField(2, required=True)