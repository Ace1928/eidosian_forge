def _unsigned_int_to_char(i):
    """
    Convert an integer 0-63 to ASCII character.
    """
    return _base64LikeEncoding[i]