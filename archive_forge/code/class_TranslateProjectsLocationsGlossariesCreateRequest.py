from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TranslateProjectsLocationsGlossariesCreateRequest(_messages.Message):
    """A TranslateProjectsLocationsGlossariesCreateRequest object.

  Fields:
    glossary: A Glossary resource to be passed as the request body.
    parent: Required. The project name.
  """
    glossary = _messages.MessageField('Glossary', 1)
    parent = _messages.StringField(2, required=True)