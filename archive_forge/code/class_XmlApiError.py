from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.api_lib.util import exceptions as api_exceptions
from googlecloudsdk.api_lib.util import resource
from googlecloudsdk.core import exceptions as core_exceptions
from six.moves import urllib
class XmlApiError(CloudApiError, api_exceptions.HttpException):
    """Translates a botocore ClientError and allows formatting.

  Attributes:
    error: The original ClientError.
    error_format: An S3ErrorPayload format string.
    payload: The S3ErrorPayload object.
  """

    def __init__(self, error, error_format='{botocore_error_string}'):
        super(XmlApiError, self).__init__(error, error_format=error_format, payload_class=S3ErrorPayload)