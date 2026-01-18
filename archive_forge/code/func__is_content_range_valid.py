import re
def _is_content_range_valid(start, stop, length, response=False):
    if (start is None) != (stop is None):
        return False
    elif start is None:
        return length is None or length >= 0
    elif length is None:
        return 0 <= start < stop
    elif start >= stop:
        return False
    elif response and stop > length:
        return False
    else:
        return 0 <= start < length