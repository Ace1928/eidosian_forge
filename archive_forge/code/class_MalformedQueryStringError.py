import sys
import six
class MalformedQueryStringError(Exception):
    """
    Query string is malformed, can't parse it :(
    """
    pass