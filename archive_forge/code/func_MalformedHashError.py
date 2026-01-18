def MalformedHashError(handler=None, reason=None):
    """error raised if recognized-but-malformed hash provided to handler"""
    text = 'malformed %s hash' % _get_name(handler)
    if reason:
        text = '%s (%s)' % (text, reason)
    return ValueError(text)