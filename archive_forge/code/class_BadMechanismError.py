import enum
import typing
class BadMechanismError(SpnegoError):
    ERROR_CODE = ErrorCode.bad_mech
    _BASE_MESSAGE = 'An unsupported mechanism was requested'
    _GSSAPI_CODE = 65536
    _SSPI_CODE = -2146893051