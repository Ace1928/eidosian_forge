from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SearchConfig(_messages.Message):
    """Contains the configuration for FHIR search.

  Fields:
    searchParameters: A list of search parameters in this FHIR store that are
      used to configure this FHIR store.
  """
    searchParameters = _messages.MessageField('SearchParameter', 1, repeated=True)