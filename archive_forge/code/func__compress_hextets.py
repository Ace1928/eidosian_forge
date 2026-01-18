import functools
@classmethod
def _compress_hextets(cls, hextets):
    """Compresses a list of hextets.

        Compresses a list of strings, replacing the longest continuous
        sequence of "0" in the list with "" and adding empty strings at
        the beginning or at the end of the string such that subsequently
        calling ":".join(hextets) will produce the compressed version of
        the IPv6 address.

        Args:
            hextets: A list of strings, the hextets to compress.

        Returns:
            A list of strings.

        """
    best_doublecolon_start = -1
    best_doublecolon_len = 0
    doublecolon_start = -1
    doublecolon_len = 0
    for index, hextet in enumerate(hextets):
        if hextet == '0':
            doublecolon_len += 1
            if doublecolon_start == -1:
                doublecolon_start = index
            if doublecolon_len > best_doublecolon_len:
                best_doublecolon_len = doublecolon_len
                best_doublecolon_start = doublecolon_start
        else:
            doublecolon_len = 0
            doublecolon_start = -1
    if best_doublecolon_len > 1:
        best_doublecolon_end = best_doublecolon_start + best_doublecolon_len
        if best_doublecolon_end == len(hextets):
            hextets += ['']
        hextets[best_doublecolon_start:best_doublecolon_end] = ['']
        if best_doublecolon_start == 0:
            hextets = [''] + hextets
    return hextets