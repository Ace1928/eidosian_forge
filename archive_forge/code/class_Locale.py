from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class Locale(_messages.Message):
    """A location/region designation for language.

  Fields:
    id: The id for this locale. Example: "en_US".
    name: A human-friendly name for this language/locale. Example: "English".
    region: A human-friendly string representing the region for this locale.
      Example: "United States". Not present for every locale.
    tags: Tags for this dimension. Example: "default".
  """
    id = _messages.StringField(1)
    name = _messages.StringField(2)
    region = _messages.StringField(3)
    tags = _messages.StringField(4, repeated=True)