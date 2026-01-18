from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CharacterMaskField(_messages.Message):
    """Replace field value with masking character. Supported
  [types](https://www.hl7.org/fhir/datatypes.html): Code, Decimal, HumanName,
  Id, LanguageCode, Markdown, Oid, String, Uri, Uuid, Xhtml.
  """