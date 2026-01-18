def _decode_flips(chars):
    """
    Read a bit field from base64-like encoding.
    """
    flips = []
    for char in chars:
        val = _char_to_unsigned_int(char)
        for j in range(6):
            flips.append(bool(val >> j & 1))
    return flips