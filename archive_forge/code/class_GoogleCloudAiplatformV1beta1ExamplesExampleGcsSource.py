from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1ExamplesExampleGcsSource(_messages.Message):
    """The Cloud Storage input instances.

  Enums:
    DataFormatValueValuesEnum: The format in which instances are given, if not
      specified, assume it's JSONL format. Currently only JSONL format is
      supported.

  Fields:
    dataFormat: The format in which instances are given, if not specified,
      assume it's JSONL format. Currently only JSONL format is supported.
    gcsSource: The Cloud Storage location for the input instances.
  """

    class DataFormatValueValuesEnum(_messages.Enum):
        """The format in which instances are given, if not specified, assume it's
    JSONL format. Currently only JSONL format is supported.

    Values:
      DATA_FORMAT_UNSPECIFIED: Format unspecified, used when unset.
      JSONL: Examples are stored in JSONL files.
    """
        DATA_FORMAT_UNSPECIFIED = 0
        JSONL = 1
    dataFormat = _messages.EnumField('DataFormatValueValuesEnum', 1)
    gcsSource = _messages.MessageField('GoogleCloudAiplatformV1beta1GcsSource', 2)