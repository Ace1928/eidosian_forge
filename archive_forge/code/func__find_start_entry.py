def _find_start_entry(line, n):
    """Find the starting character for entry ``n`` in a space delimited ``line`` (PRIVATE).

    n is counted starting with 1.
    The n=1 field by definition begins at the first character.

    Returns
    -------
    starting character : str
        The starting character for entry ``n``.

    """
    if n == 1:
        return 0
    c = 1
    leng = len(line)
    if line[0] == ' ':
        infield = False
        field = 0
    else:
        infield = True
        field = 1
    while c < leng and field < n:
        if infield:
            if line[c] == ' ' and line[c - 1] != ' ':
                infield = False
            elif line[c] != ' ':
                infield = True
                field += 1
        c += 1
    return c - 1