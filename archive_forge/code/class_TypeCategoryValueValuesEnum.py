from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TypeCategoryValueValuesEnum(_messages.Enum):
    """The class of identifiers where this infoType belongs

    Values:
      TYPE_UNSPECIFIED: Unused type
      PII: Personally identifiable information, for example, a name or phone
        number
      SPII: Personally identifiable information that is especially sensitive,
        for example, a passport number.
      DEMOGRAPHIC: Attributes that can partially identify someone, especially
        in combination with other attributes, like age, height, and gender.
      CREDENTIAL: Confidential or secret information, for example, a password.
      GOVERNMENT_ID: An identification document issued by a government.
      DOCUMENT: A document, for example, a resume or source code.
      CONTEXTUAL_INFORMATION: Information that is not sensitive on its own,
        but provides details about the circumstances surrounding an entity or
        an event.
    """
    TYPE_UNSPECIFIED = 0
    PII = 1
    SPII = 2
    DEMOGRAPHIC = 3
    CREDENTIAL = 4
    GOVERNMENT_ID = 5
    DOCUMENT = 6
    CONTEXTUAL_INFORMATION = 7