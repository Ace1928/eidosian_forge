from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OutputDataFormatValueValuesEnum(_messages.Enum):
    """Optional. Format of the output data files, defaults to JSON.

    Values:
      DATA_FORMAT_UNSPECIFIED: Unspecified format.
      JSON: Each line of the file is a JSON dictionary representing one
        record.
      TEXT: Deprecated. Use JSON instead.
      TF_RECORD: The source file is a TFRecord file. Currently available only
        for input data.
      TF_RECORD_GZIP: The source file is a GZIP-compressed TFRecord file.
        Currently available only for input data.
      FILE_LIST: Each line of the file is the location of an instance to
        process. Currently available only for input data.
      CSV: Values are comma-separated rows, with keys in a separate file.
        Currently available only for output data.
    """
    DATA_FORMAT_UNSPECIFIED = 0
    JSON = 1
    TEXT = 2
    TF_RECORD = 3
    TF_RECORD_GZIP = 4
    FILE_LIST = 5
    CSV = 6