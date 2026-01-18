from collections import namedtuple
import re
import textwrap
import warnings
def accept_property():
    doc = '\n        Property representing the ``Accept`` header.\n\n        (:rfc:`RFC 7231, section 5.3.2 <7231#section-5.3.2>`)\n\n        The header value in the request environ is parsed and a new object\n        representing the header is created every time we *get* the value of the\n        property. (*set* and *del* change the header value in the request\n        environ, and do not involve parsing.)\n    '
    ENVIRON_KEY = 'HTTP_ACCEPT'

    def fget(request):
        """Get an object representing the header in the request."""
        return create_accept_header(header_value=request.environ.get(ENVIRON_KEY))

    def fset(request, value):
        """
        Set the corresponding key in the request environ.

        `value` can be:

        * ``None``
        * a ``str`` header value
        * a ``dict``, with media ranges ``str``'s (including any media type
          parameters) as keys, and either qvalues ``float``'s or (*qvalues*,
          *extension_params*) tuples as values, where *extension_params* is a
          ``str`` of the extension parameters segment of the header element,
          starting with the first '``;``'
        * a ``tuple`` or ``list``, where each item is either a header element
          ``str``, or a (*media_range*, *qvalue*, *extension_params*) ``tuple``
          or ``list`` where *media_range* is a ``str`` of the media range
          including any media type parameters, and *extension_params* is a
          ``str`` of the extension parameters segment of the header element,
          starting with the first '``;``'
        * an :class:`AcceptValidHeader`, :class:`AcceptNoHeader`, or
          :class:`AcceptInvalidHeader` instance
        * object of any other type that returns a value for ``__str__``
        """
        if value is None or isinstance(value, AcceptNoHeader):
            fdel(request=request)
        else:
            if isinstance(value, (AcceptValidHeader, AcceptInvalidHeader)):
                header_value = value.header_value
            else:
                header_value = Accept._python_value_to_header_str(value=value)
            request.environ[ENVIRON_KEY] = header_value

    def fdel(request):
        """Delete the corresponding key from the request environ."""
        try:
            del request.environ[ENVIRON_KEY]
        except KeyError:
            pass
    return property(fget, fset, fdel, textwrap.dedent(doc))