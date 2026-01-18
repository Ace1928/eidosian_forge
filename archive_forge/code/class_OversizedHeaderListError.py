class OversizedHeaderListError(HPACKDecodingError):
    """
    A header list that was larger than we allow has been received. This may be
    a DoS attack.

    .. versionadded:: 2.3.0
    """
    pass