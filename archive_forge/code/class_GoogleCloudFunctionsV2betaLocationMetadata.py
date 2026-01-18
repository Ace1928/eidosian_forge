from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudFunctionsV2betaLocationMetadata(_messages.Message):
    """Extra GCF specific location information.

  Enums:
    EnvironmentsValueListEntryValuesEnum:

  Fields:
    environments: The Cloud Function environments this location supports.
  """

    class EnvironmentsValueListEntryValuesEnum(_messages.Enum):
        """EnvironmentsValueListEntryValuesEnum enum type.

    Values:
      ENVIRONMENT_UNSPECIFIED: Unspecified
      GEN_1: Gen 1
      GEN_2: Gen 2
    """
        ENVIRONMENT_UNSPECIFIED = 0
        GEN_1 = 1
        GEN_2 = 2
    environments = _messages.EnumField('EnvironmentsValueListEntryValuesEnum', 1, repeated=True)