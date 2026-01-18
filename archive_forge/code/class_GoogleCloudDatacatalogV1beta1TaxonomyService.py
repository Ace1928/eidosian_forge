from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDatacatalogV1beta1TaxonomyService(_messages.Message):
    """The source system of the Taxonomy.

  Enums:
    NameValueValuesEnum: The Google Cloud service name.

  Fields:
    identity: The service agent for the service.
    name: The Google Cloud service name.
  """

    class NameValueValuesEnum(_messages.Enum):
        """The Google Cloud service name.

    Values:
      MANAGING_SYSTEM_UNSPECIFIED: Default value
      MANAGING_SYSTEM_DATAPLEX: Dataplex.
      MANAGING_SYSTEM_OTHER: Other
    """
        MANAGING_SYSTEM_UNSPECIFIED = 0
        MANAGING_SYSTEM_DATAPLEX = 1
        MANAGING_SYSTEM_OTHER = 2
    identity = _messages.StringField(1)
    name = _messages.EnumField('NameValueValuesEnum', 2)