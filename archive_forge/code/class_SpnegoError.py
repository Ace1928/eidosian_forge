import enum
import typing
class SpnegoError(Exception, metaclass=_SpnegoErrorRegistry):
    """Common error for SPNEGO exception.

    Creates an common error record for SPNEGO errors raised by pyspnego. This error record can wrap system level error
    records raised by GSSAPI or SSPI and wrap them into a common error record across the various platforms. While this
    reflects the GSSAPI major codes that can be raised, it is up to the GSSAPI platform to conform to those error
    codes. Some platforms like MIT krb5 always report `GSS_S_FAILURE` and use the minor code to report the actual
    error message.

    Args:
        error_code: The ErrorCode for the error, this must be set if base_error is not set.
        base_error: The system level error from SSPI or GSSAPI, this must be set if error_code is not set.
        context_msg: Optional message to provide more context around the error.

    Attributes:
        base_error (Optional[Union[GSSError, WinError]]): The system level error if one was provided.
    """

    def __init__(self, error_code: typing.Optional[typing.Union[int, ErrorCode]]=None, base_error: typing.Optional[Exception]=None, context_msg: typing.Optional[str]=None) -> None:
        self.base_error = base_error
        self._error_code = error_code
        self._context_message = context_msg
        super(SpnegoError, self).__init__(self.message)

    @property
    def nt_status(self) -> int:
        """The Windows NT Status code that represents this error."""
        codes = getattr(self, '_SSPI_CODE', self._error_code) or 4294967295
        return codes[0] if isinstance(codes, (list, tuple)) else codes

    @property
    def message(self) -> str:
        error_code = self._error_code if self._error_code is not None else 4294967295
        if self.base_error:
            base_message = str(self.base_error)
        else:
            base_message = getattr(self, '_BASE_MESSAGE', 'Unknown error code')
        msg = 'SpnegoError (%d): %s' % (error_code, base_message)
        if self._context_message:
            msg += ', Context: %s' % self._context_message
        return msg