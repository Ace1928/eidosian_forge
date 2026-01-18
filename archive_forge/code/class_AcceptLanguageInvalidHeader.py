from collections import namedtuple
import re
import textwrap
import warnings
class AcceptLanguageInvalidHeader(_AcceptLanguageInvalidOrNoHeader):
    """
    Represent an invalid ``Accept-Language`` header.

    An invalid header is one that does not conform to
    :rfc:`7231#section-5.3.5`. As specified in the RFC, an empty header is an
    invalid ``Accept-Language`` header.

    :rfc:`7231` does not provide any guidance on what should happen if the
    ``Accept-Language`` header has an invalid value. This implementation
    disregards the header, and treats it as if there is no ``Accept-Language``
    header in the request.

    This object should not be modified. To add to the header, we can use the
    addition operators (``+`` and ``+=``), which return a new object (see the
    docstring for :meth:`AcceptLanguageInvalidHeader.__add__`).
    """

    def __init__(self, header_value):
        """
        Create an :class:`AcceptLanguageInvalidHeader` instance.
        """
        self._header_value = header_value
        self._parsed = None
        self._parsed_nonzero = None

    def copy(self):
        """
        Create a copy of the header object.

        """
        return self.__class__(self._header_value)

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

    def __add__(self, other):
        """
        Add to header, creating a new header object.

        `other` can be:

        * ``None``
        * a ``str``
        * a ``dict``, with language ranges as keys and qvalues as values
        * a ``tuple`` or ``list``, of language range ``str``'s or of ``tuple``
          or ``list`` (language range, qvalue) pairs (``str``'s and pairs can
          be mixed within the ``tuple`` or ``list``)
        * an :class:`AcceptLanguageValidHeader`,
          :class:`AcceptLanguageNoHeader`, or
          :class:`AcceptLanguageInvalidHeader` instance
        * object of any other type that returns a value for ``__str__``

        If `other` is a valid header value or an
        :class:`AcceptLanguageValidHeader` instance, a new
        :class:`AcceptLanguageValidHeader` instance with the valid header value
        is returned.

        If `other` is ``None``, an :class:`AcceptLanguageNoHeader` instance, an
        invalid header value, or an :class:`AcceptLanguageInvalidHeader`
        instance, a new :class:`AcceptLanguageNoHeader` instance is returned.
        """
        if isinstance(other, AcceptLanguageValidHeader):
            return AcceptLanguageValidHeader(header_value=other.header_value)
        if isinstance(other, (AcceptLanguageNoHeader, AcceptLanguageInvalidHeader)):
            return AcceptLanguageNoHeader()
        return self._add_instance_and_non_accept_language_type(instance=self, other=other)

    def __radd__(self, other):
        """
        Add to header, creating a new header object.

        See the docstring for :meth:`AcceptLanguageValidHeader.__add__`.
        """
        return self._add_instance_and_non_accept_language_type(instance=self, other=other, instance_on_the_right=True)

    def __repr__(self):
        return '<{}>'.format(self.__class__.__name__)

    def __str__(self):
        """Return the ``str`` ``'<invalid header value>'``."""
        return '<invalid header value>'

    def _add_instance_and_non_accept_language_type(self, instance, other, instance_on_the_right=False):
        if not other:
            return AcceptLanguageNoHeader()
        other_header_value = self._python_value_to_header_str(value=other)
        try:
            return AcceptLanguageValidHeader(header_value=other_header_value)
        except ValueError:
            return AcceptLanguageNoHeader()