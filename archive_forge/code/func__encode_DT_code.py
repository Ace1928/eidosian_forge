def _encode_DT_code(code):
    """
    Given a DT code, convert to base64-like encoding.
    """
    num_chars = _get_num_chars(code)
    chars = _unsigned_int_to_char(num_chars + 52)
    chars += _int_to_chars(len(code), num_chars)
    for comp in code:
        chars += _int_to_chars(len(comp), num_chars)
        for crossing in comp:
            chars += _int_to_chars(crossing, num_chars)
    return chars