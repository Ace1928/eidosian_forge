import enum
import typing
class BadBindingsError(SpnegoError):
    ERROR_CODE = ErrorCode.bad_bindings
    _BASE_MESSAGE = 'Invalid channel bindings'
    _GSSAPI_CODE = 262144
    _SSPI_CODE = -2146892986