from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RequestModelValueValuesEnum(_messages.Enum):
    """Specifies the API request model used to call the storage service. When
    not specified, the default value of RequestModel
    REQUEST_MODEL_VIRTUAL_HOSTED_STYLE is used.

    Values:
      REQUEST_MODEL_UNSPECIFIED: RequestModel is not specified.
      REQUEST_MODEL_VIRTUAL_HOSTED_STYLE: Perform requests using Virtual
        Hosted Style. Example: https://bucket-
        name.s3.region.amazonaws.com/key-name
      REQUEST_MODEL_PATH_STYLE: Perform requests using Path Style. Example:
        https://s3.region.amazonaws.com/bucket-name/key-name
    """
    REQUEST_MODEL_UNSPECIFIED = 0
    REQUEST_MODEL_VIRTUAL_HOSTED_STYLE = 1
    REQUEST_MODEL_PATH_STYLE = 2