def _consume_int_and_advance(chars, pos, num_chars):
    """
    Read the num_chars characters from the string chars and interpret it as
    base64-like encoded integer. Also return the end of that integer. This
    is supposed to be in a pattern like this::

        >>> chars = "abcdef"
        >>> pos = 0 # Start at the beginning
        >>> first_integer,  pos = _consume_int_and_advance(chars, pos, 2)
        >>> second_integer, pos = _consume_int_and_advance(chars, pos, 2)
        >>> third_integer,  pos = _consume_int_and_advance(chars, pos, 2)
    """
    end = pos + num_chars
    return (_chars_to_int(chars[pos:end]), end)