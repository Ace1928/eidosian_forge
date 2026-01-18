def acceptable_content_type(accept_header, content_types, ignore_wildcard=True):
    """Determines if the given content type is acceptable to the user agent.

    The accept_header should be the value present in the HTTP
    "Accept:" header.  In mod_python this is typically obtained from
    the req.http_headers_in table; in WSGI it is environ["Accept"];
    other web frameworks may provide other methods of obtaining it.

    Optionally the accept_header parameter can be pre-parsed, as
    returned from the parse_accept_header() function in this module.

    The content_types argument should either be a single MIME media
    type string, or a sequence of them.  It represents the set of
    content types that the caller (server) is willing to send.
    Generally, the server content_types should not contain any
    wildcarded values.

    This function determines which content type which is the most
    preferred and is acceptable to both the user agent and the server.
    If one is negotiated it will return a four-valued tuple like:

        (server_content_type, ua_content_range, qvalue, accept_parms)

    The first tuple value is one of the server's content_types, while
    the remaining tuple values descript which of the client's
    acceptable content_types was matched.  In most cases accept_parms
    will be an empty list (see description of parse_accept_header()
    for more details).

    If no content type could be negotiated, then this function will
    return None (and the caller should typically cause an HTTP 406 Not
    Acceptable as a response).

    Note that the wildcarded content type "*/*" sent by the client
    will be ignored, since it is often incorrectly sent by web
    browsers that don't really mean it.  To override this, call with
    ignore_wildcard=False.  Partial wildcards such as "image/*" will
    always be processed, but be at a lower priority than a complete
    matching type.

    See also: RFC 2616 section 14.1, and
    <http://www.iana.org/assignments/media-types/>

    """
    if _is_string(accept_header):
        accept_list = parse_accept_header(accept_header)
    else:
        accept_list = accept_header
    if _is_string(content_types):
        content_types = [content_types]
    server_ctlist = [content_type(ct) for ct in content_types]
    del ct
    best = None
    for server_ct in server_ctlist:
        best_for_this = None
        for client_ct, qvalue, aargs in accept_list:
            if ignore_wildcard and client_ct.is_universal_wildcard():
                continue
            matchlen = 0
            if client_ct.is_universal_wildcard():
                matchlen = 1
            elif client_ct.major == server_ct.major:
                if client_ct.minor == '*':
                    matchlen = 2
                elif client_ct.minor == server_ct.minor:
                    matchlen = 3
                    for pname, pval in client_ct.parmdict.items():
                        sval = server_ct.parmdict.get(pname)
                        if pname == 'charset':
                            pval = canonical_charset(pval)
                            sval = canonical_charset(sval)
                        if sval == pval:
                            matchlen = matchlen + 1
                        else:
                            matchlen = 0
                            break
                else:
                    matchlen = 0
            if matchlen > 0:
                if not best_for_this or matchlen > best_for_this[-1] or (matchlen == best_for_this[-1] and qvalue > best_for_this[2]):
                    best_for_this = (server_ct, client_ct, qvalue, aargs, matchlen)
        if not best or (best_for_this and best_for_this[2] > best[2]):
            best = best_for_this
    if not best or best[1] <= 0:
        return None
    return best[:-1]