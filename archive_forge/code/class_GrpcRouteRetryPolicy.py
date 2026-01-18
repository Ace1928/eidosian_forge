from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GrpcRouteRetryPolicy(_messages.Message):
    """The specifications for retries.

  Fields:
    numRetries: Specifies the allowed number of retries. This number must be >
      0. If not specified, default to 1.
    retryConditions: - connect-failure: Router will retry on failures
      connecting to Backend Services, for example due to connection timeouts.
      - refused-stream: Router will retry if the backend service resets the
      stream with a REFUSED_STREAM error code. This reset type indicates that
      it is safe to retry. - cancelled: Router will retry if the gRPC status
      code in the response header is set to cancelled - deadline-exceeded:
      Router will retry if the gRPC status code in the response header is set
      to deadline-exceeded - resource-exhausted: Router will retry if the gRPC
      status code in the response header is set to resource-exhausted -
      unavailable: Router will retry if the gRPC status code in the response
      header is set to unavailable
  """
    numRetries = _messages.IntegerField(1, variant=_messages.Variant.UINT32)
    retryConditions = _messages.StringField(2, repeated=True)