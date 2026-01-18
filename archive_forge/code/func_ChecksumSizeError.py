def ChecksumSizeError(handler, raw=False):
    """error raised if hash was recognized, but checksum was wrong size"""
    checksum_size = handler.checksum_size
    unit = 'bytes' if raw else 'chars'
    reason = 'checksum must be exactly %d %s' % (checksum_size, unit)
    return MalformedHashError(handler, reason)