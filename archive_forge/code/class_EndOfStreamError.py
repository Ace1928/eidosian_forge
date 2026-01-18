class EndOfStreamError(SubstrateUnderrunError):
    """ASN.1 data structure deserialization error

    The `EndOfStreamError` exception indicates the condition of the input
    stream has been closed.
    """