import functools
@classmethod
def _parse_octet(cls, octet_str):
    """Convert a decimal octet into an integer.

        Args:
            octet_str: A string, the number to parse.

        Returns:
            The octet as an integer.

        Raises:
            ValueError: if the octet isn't strictly a decimal from [0..255].

        """
    if not octet_str:
        raise ValueError('Empty octet not permitted')
    if not (octet_str.isascii() and octet_str.isdigit()):
        msg = 'Only decimal digits permitted in %r'
        raise ValueError(msg % octet_str)
    if len(octet_str) > 3:
        msg = 'At most 3 characters permitted in %r'
        raise ValueError(msg % octet_str)
    if octet_str != '0' and octet_str[0] == '0':
        msg = 'Leading zeros are not permitted in %r'
        raise ValueError(msg % octet_str)
    octet_int = int(octet_str, 10)
    if octet_int > 255:
        raise ValueError('Octet %d (> 255) not permitted' % octet_int)
    return octet_int