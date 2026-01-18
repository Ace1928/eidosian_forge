import bisect
def _encode_range(start, end):
    return start << 32 | end