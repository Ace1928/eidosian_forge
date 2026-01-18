from collections import namedtuple
import re
import textwrap
import warnings
def accept_encoding_property():
    doc = '\n        Property representing the ``Accept-Encoding`` header.\n\n        (:rfc:`RFC 7231, section 5.3.4 <7231#section-5.3.4>`)\n\n        The header value in the request environ is parsed and a new object\n        representing the header is created every time we *get* the value of the\n        property. (*set* and *del* change the header value in the request\n        environ, and do not involve parsing.)\n    '
    ENVIRON_KEY = 'HTTP_ACCEPT_ENCODING'

    def fget(request):
        """Get an object representing the header in the request."""
        return create_accept_encoding_header(header_value=request.environ.get(ENVIRON_KEY))

    def fset(request, value):
        """
        Set the corresponding key in the request environ.

        `value` can be:

        * ``None``
        * a ``str`` header value
        * a ``dict``, with content-coding, ``identity`` or ``*`` ``str``'s as
          keys, and qvalue ``float``'s as values
        * a ``tuple`` or ``list``, where each item is either a header element
          ``str``, or a (content-coding/``identity``/``*``, qvalue) ``tuple``
          or ``list``
        * an :class:`AcceptEncodingValidHeader`,
          :class:`AcceptEncodingNoHeader`, or
          :class:`AcceptEncodingInvalidHeader` instance
        * object of any other type that returns a value for ``__str__``
        """
        if value is None or isinstance(value, AcceptEncodingNoHeader):
            fdel(request=request)
        else:
            if isinstance(value, (AcceptEncodingValidHeader, AcceptEncodingInvalidHeader)):
                header_value = value.header_value
            else:
                header_value = AcceptEncoding._python_value_to_header_str(value=value)
            request.environ[ENVIRON_KEY] = header_value

    def fdel(request):
        """Delete the corresponding key from the request environ."""
        try:
            del request.environ[ENVIRON_KEY]
        except KeyError:
            pass
    return property(fget, fset, fdel, textwrap.dedent(doc))