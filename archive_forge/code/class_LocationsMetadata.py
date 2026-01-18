from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LocationsMetadata(_messages.Message):
    """Main metadata for the Locations API for STT V2. Currently this is just
  the metadata about locales, models, and features

  Fields:
    accessMetadata: Information about access metadata for the region and given
      project.
    languages: Information about available locales, models, and features
      represented in the hierarchical structure of locales -> models ->
      features
  """
    accessMetadata = _messages.MessageField('AccessMetadata', 1)
    languages = _messages.MessageField('LanguageMetadata', 2)