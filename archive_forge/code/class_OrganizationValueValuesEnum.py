from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OrganizationValueValuesEnum(_messages.Enum):
    """For whom the client library is being published.

    Values:
      CLIENT_LIBRARY_ORGANIZATION_UNSPECIFIED: Not useful.
      CLOUD: Google Cloud Platform Org.
      ADS: Ads (Advertising) Org.
      PHOTOS: Photos Org.
      STREET_VIEW: Street View Org.
      SHOPPING: Shopping Org.
      GEO: Geo Org.
      GENERATIVE_AI: Generative AI - https://developers.generativeai.google
    """
    CLIENT_LIBRARY_ORGANIZATION_UNSPECIFIED = 0
    CLOUD = 1
    ADS = 2
    PHOTOS = 3
    STREET_VIEW = 4
    SHOPPING = 5
    GEO = 6
    GENERATIVE_AI = 7