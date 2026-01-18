import enum
import typing
class BadNameError(SpnegoError):
    ERROR_CODE = ErrorCode.bad_name
    _BASE_MESSAGE = 'An invalid name was supplied'
    _GSSAPI_CODE = 1310722
    _SSPI_CODE = -2146893053