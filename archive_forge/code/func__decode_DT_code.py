def _decode_DT_code(chars):
    """
    Given a base64-like encoding, return the DT code and the position where
    the remaining characters not consumed yet start.
    """
    code = []
    num_chars = _char_to_unsigned_int(chars[0]) - 52
    num_components, pos = _consume_int_and_advance(chars, 1, num_chars)
    for i in range(num_components):
        component = []
        num_crossings, pos = _consume_int_and_advance(chars, pos, num_chars)
        for j in range(num_crossings):
            crossing, pos = _consume_int_and_advance(chars, pos, num_chars)
            component.append(crossing)
        code.append(tuple(component))
    return (code, pos)