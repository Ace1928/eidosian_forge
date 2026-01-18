import enum
import typing
class NoContextError(SpnegoError):
    ERROR_CODE = ErrorCode.no_context
    _BASE_MESSAGE = 'No context has been established, or invalid handled passed in'
    _GSSAPI_CODE = 524288
    _SSPI_CODE = -2146893055