import enum
import typing
class OperationNotAvailableError(SpnegoError):
    ERROR_CODE = ErrorCode.unavailable
    _BASE_MESSAGE = 'Operation not supported or available'
    _GSSAPI_CODE = 1048576
    _SSPI_CODE = -2146893054