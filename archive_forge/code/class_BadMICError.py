import enum
import typing
class BadMICError(SpnegoError):
    ERROR_CODE = ErrorCode.bad_mic
    _BASE_MESSAGE = 'A token had an invalid Message Integrity Check (MIC)'
    _GSSAPI_CODE = 3932166
    _SSPI_CODE = -2146893041