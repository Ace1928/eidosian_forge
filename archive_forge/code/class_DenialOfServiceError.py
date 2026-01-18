import h2.errors
class DenialOfServiceError(ProtocolError):
    """
    Emitted when the remote peer exhibits a behaviour that is likely to be an
    attempt to perform a Denial of Service attack on the implementation. This
    is a form of ProtocolError that carries a different error code, and allows
    more easy detection of this kind of behaviour.

    .. versionadded:: 2.5.0
    """
    error_code = h2.errors.ErrorCodes.ENHANCE_YOUR_CALM