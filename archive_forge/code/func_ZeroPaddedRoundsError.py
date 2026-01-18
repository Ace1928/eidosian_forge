def ZeroPaddedRoundsError(handler=None):
    """error raised if hash was recognized but contained zero-padded rounds field"""
    return MalformedHashError(handler, 'zero-padded rounds')