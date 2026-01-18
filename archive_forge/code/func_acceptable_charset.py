def acceptable_charset(accept_charset_header, charsets, ignore_wildcard=True, default='ISO-8859-1'):
    """
    Determines if the given charset is acceptable to the user agent.

    The accept_charset_header should be the value present in the HTTP
    "Accept-Charset:" header.  In mod_python this is typically
    obtained from the req.http_headers table; in WSGI it is
    environ["Accept-Charset"]; other web frameworks may provide other
    methods of obtaining it.

    Optionally the accept_charset_header parameter can instead be the
    list returned from the parse_accept_header() function in this
    module.

    The charsets argument should either be a charset identifier string,
    or a sequence of them.

    This function returns the charset identifier string which is the
    most prefered and is acceptable to both the user agent and the
    caller.  It will return the default value if no charset is negotiable.
    
    Note that the wildcarded charset "*" will be ignored.  To override
    this, call with ignore_wildcard=False.

    See also: RFC 2616 section 14.2, and
    <http://www.iana.org/assignments/character-sets>

    """
    if default:
        default = canonical_charset(default)
    if _is_string(accept_charset_header):
        accept_list = parse_accept_header(accept_charset_header)
    else:
        accept_list = accept_charset_header
    if _is_string(charsets):
        charsets = [canonical_charset(charsets)]
    else:
        charsets = [canonical_charset(c) for c in charsets]
    best = None
    for c, qvalue, _junk in accept_list:
        if c == '*':
            default = None
            if ignore_wildcard:
                continue
            if not best or qvalue > best[1]:
                best = (c, qvalue)
        else:
            c = canonical_charset(c)
            for test_c in charsets:
                if c == default:
                    default = None
                if c == test_c and (not best or best[0] == '*' or qvalue > best[1]):
                    best = (c, qvalue)
    if default and default in [test_c.upper() for test_c in charsets]:
        best = (default, 1)
    if best[0] == '*':
        best = (charsets[0], best[1])
    return best