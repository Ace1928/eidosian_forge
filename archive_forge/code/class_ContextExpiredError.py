import enum
import typing
class ContextExpiredError(SpnegoError):
    ERROR_CODE = ErrorCode.context_expired
    _BASE_MESSAGE = 'Security context has expired'
    _GSSAPI_CODE = 786432
    _SSPI_CODE = -2146893033