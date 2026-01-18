def _default_wrap(indent):
    """Return default wrap rule for _wrap_kegg (PRIVATE).

    A wrap rule is a list with the following elements:
    [indent, connect, (splitstr, connect, splitafter, keep), ...]
    """
    return [indent, '', (' ', '', 1, 0)]