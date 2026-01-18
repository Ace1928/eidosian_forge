class NoTokenLookupException(Exception):
    """This form of authentication does not support looking up
       endpoints from an existing token.
    """
    pass