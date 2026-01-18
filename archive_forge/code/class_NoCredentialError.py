import enum
import typing
class NoCredentialError(SpnegoError):
    ERROR_CODE = ErrorCode.no_cred
    _BASE_MESSAGE = 'No credentials were supplied, or the credentials were unavailable or inaccessible'
    _GSSAPI_CODE = 458752
    _SSPI_CODE = -2146893042