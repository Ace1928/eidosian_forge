import enum
import typing
class UnsupportedQop(SpnegoError):
    ERROR_CODE = ErrorCode.bad_qop
    _BASE_MESSAGE = 'The quality-of-protection requested could not be provided'
    _GSSAPI_CODE = 917504
    _SSPI_CODE = -2146893046