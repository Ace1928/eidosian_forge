import enum
import typing
class InvalidCredentialError(SpnegoError):
    ERROR_CODE = ErrorCode.invalid_credential
    _BASE_MESSAGE = 'A credential was invalid'
    _GSSAPI_CODE = 655360
    _SSPI_CODE = -1073741715