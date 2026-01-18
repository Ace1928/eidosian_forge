from collections import namedtuple
import re
import textwrap
import warnings
class AcceptCharsetInvalidHeader(_AcceptCharsetInvalidOrNoHeader):
    """
    Represent an invalid ``Accept-Charset`` header.

    An invalid header is one that does not conform to
    :rfc:`7231#section-5.3.3`. As specified in the RFC, an empty header is an
    invalid ``Accept-Charset`` header.

    :rfc:`7231` does not provide any guidance on what should happen if the
    ``Accept-Charset`` header has an invalid value. This implementation
    disregards the header, and treats it as if there is no ``Accept-Charset``
    header in the request.

    This object should not be modified. To add to the header, we can use the
    addition operators (``+`` and ``+=``), which return a new object (see the
    docstring for :meth:`AcceptCharsetInvalidHeader.__add__`).
    """

    @property
    def header_value(self):
        """(``str`` or ``None``) The header value."""
        return self._header_value

    @property
    def parsed(self):
        """
        (``list`` or ``None``) Parsed form of the header.

        As the header is invalid and cannot be parsed, this is ``None``.
        """
        return self._parsed

    def __init__(self, header_value):
        """
        Create an :class:`AcceptCharsetInvalidHeader` instance.
        """
        self._header_value = header_value
        self._parsed = None
        self._parsed_nonzero = None

    def copy(self):
        """
        Create a copy of the header object.

        """
        return self.__class__(self._header_value)

    def __add__(self, other):
        """
        Add to header, creating a new header object.

        `other` can be:

        * ``None``
        * a ``str`` header value
        * a ``dict``, where keys are charsets and values are qvalues
        * a ``tuple`` or ``list``, where each item is a charset ``str`` or a
          ``tuple`` or ``list`` (charset, qvalue) pair (``str``'s and pairs
          can be mixed within the ``tuple`` or ``list``)
        * an :class:`AcceptCharsetValidHeader`, :class:`AcceptCharsetNoHeader`,
          or :class:`AcceptCharsetInvalidHeader` instance
        * object of any other type that returns a value for ``__str__``

        If `other` is a valid header value or an
        :class:`AcceptCharsetValidHeader` instance, a new
        :class:`AcceptCharsetValidHeader` instance with the valid header value
        is returned.

        If `other` is ``None``, an :class:`AcceptCharsetNoHeader` instance, an
        invalid header value, or an :class:`AcceptCharsetInvalidHeader`
        instance, a new :class:`AcceptCharsetNoHeader` instance is returned.
        """
        if isinstance(other, AcceptCharsetValidHeader):
            return AcceptCharsetValidHeader(header_value=other.header_value)
        if isinstance(other, (AcceptCharsetNoHeader, AcceptCharsetInvalidHeader)):
            return AcceptCharsetNoHeader()
        return self._add_instance_and_non_accept_charset_type(instance=self, other=other)

    def __radd__(self, other):
        """
        Add to header, creating a new header object.

        See the docstring for :meth:`AcceptCharsetValidHeader.__add__`.
        """
        return self._add_instance_and_non_accept_charset_type(instance=self, other=other, instance_on_the_right=True)

    def __repr__(self):
        return '<{}>'.format(self.__class__.__name__)

    def __str__(self):
        """Return the ``str`` ``'<invalid header value>'``."""
        return '<invalid header value>'

    def _add_instance_and_non_accept_charset_type(self, instance, other, instance_on_the_right=False):
        if not other:
            return AcceptCharsetNoHeader()
        other_header_value = self._python_value_to_header_str(value=other)
        try:
            return AcceptCharsetValidHeader(header_value=other_header_value)
        except ValueError:
            return AcceptCharsetNoHeader()