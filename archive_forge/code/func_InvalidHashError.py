def InvalidHashError(handler=None):
    """error raised if unrecognized hash provided to handler"""
    return ValueError('not a valid %s hash' % _get_name(handler))