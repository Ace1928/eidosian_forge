from collections import namedtuple
import re
import textwrap
import warnings
class AcceptCharsetValidHeader(AcceptCharset):
    """
    Represent a valid ``Accept-Charset`` header.

    A valid header is one that conforms to :rfc:`RFC 7231, section 5.3.3
    <7231#section-5.3.3>`.

    This object should not be modified. To add to the header, we can use the
    addition operators (``+`` and ``+=``), which return a new object (see the
    docstring for :meth:`AcceptCharsetValidHeader.__add__`).
    """

    @property
    def header_value(self):
        """(``str``) The header value."""
        return self._header_value

    @property
    def parsed(self):
        """
        (``list``) Parsed form of the header.

        A list of (charset, quality value) tuples.
        """
        return self._parsed

    def __init__(self, header_value):
        """
        Create an :class:`AcceptCharsetValidHeader` instance.

        :param header_value: (``str``) header value.
        :raises ValueError: if `header_value` is an invalid value for an
                            ``Accept-Charset`` header.
        """
        self._header_value = header_value
        self._parsed = list(self.parse(header_value))
        self._parsed_nonzero = [item for item in self.parsed if item[1]]

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

        If `other` is a valid header value or another
        :class:`AcceptCharsetValidHeader` instance, the two header values are
        joined with ``', '``, and a new :class:`AcceptCharsetValidHeader`
        instance with the new header value is returned.

        If `other` is ``None``, an :class:`AcceptCharsetNoHeader` instance, an
        invalid header value, or an :class:`AcceptCharsetInvalidHeader`
        instance, a new :class:`AcceptCharsetValidHeader` instance with the
        same header value as ``self`` is returned.
        """
        if isinstance(other, AcceptCharsetValidHeader):
            return create_accept_charset_header(header_value=self.header_value + ', ' + other.header_value)
        if isinstance(other, (AcceptCharsetNoHeader, AcceptCharsetInvalidHeader)):
            return self.__class__(header_value=self.header_value)
        return self._add_instance_and_non_accept_charset_type(instance=self, other=other)

    def __bool__(self):
        """
        Return whether ``self`` represents a valid ``Accept-Charset`` header.

        Return ``True`` if ``self`` represents a valid header, and ``False`` if
        it represents an invalid header, or the header not being in the
        request.

        For this class, it always returns ``True``.
        """
        return True
    __nonzero__ = __bool__

    def __contains__(self, offer):
        """
        Return ``bool`` indicating whether `offer` is acceptable.

        .. warning::

           The behavior of :meth:`AcceptCharsetValidHeader.__contains__` is
           currently being maintained for backward compatibility, but it will
           change in the future to better conform to the RFC.

        :param offer: (``str``) charset offer
        :return: (``bool``) Whether ``offer`` is acceptable according to the
                 header.

        This does not fully conform to :rfc:`RFC 7231, section 5.3.3
        <7231#section-5.3.3>`: it incorrect interprets ``*`` to mean 'match any
        charset in the header', rather than 'match any charset that is not
        mentioned elsewhere in the header'::

            >>> 'UTF-8' in AcceptCharsetValidHeader('UTF-8;q=0, *')
            True
        """
        warnings.warn('The behavior of AcceptCharsetValidHeader.__contains__ is currently being maintained for backward compatibility, but it will change in the future to better conform to the RFC.', DeprecationWarning)
        for mask, quality in self._parsed_nonzero:
            if self._old_match(mask, offer):
                return True
        return False

    def __iter__(self):
        """
        Return all the items with non-0 qvalues, in order of preference.

        .. warning::

           The behavior of this method is currently maintained for backward
           compatibility, but will change in the future.

        :return: iterator of all the items (charset or ``*``) in the header
                 with non-0 qvalues, in descending order of qvalue. If two
                 items have the same qvalue, they are returned in the order of
                 their positions in the header, from left to right.

        Please note that this is a simple filter for the items in the header
        with non-0 qvalues, and is not necessarily the same as what the client
        prefers, e.g. ``'utf-7;q=0, *'`` means 'everything but utf-7', but
        ``list(instance)`` would return only ``['*']``.
        """
        warnings.warn('The behavior of AcceptCharsetValidHeader.__iter__ is currently maintained for backward compatibility, but will change in the future.', DeprecationWarning)
        for m, q in sorted(self._parsed_nonzero, key=lambda i: i[1], reverse=True):
            yield m

    def __radd__(self, other):
        """
        Add to header, creating a new header object.

        See the docstring for :meth:`AcceptCharsetValidHeader.__add__`.
        """
        return self._add_instance_and_non_accept_charset_type(instance=self, other=other, instance_on_the_right=True)

    def __repr__(self):
        return '<{} ({!r})>'.format(self.__class__.__name__, str(self))

    def __str__(self):
        """
        Return a tidied up version of the header value.

        e.g. If the ``header_value`` is ``', \\t,iso-8859-5;q=0.000 \\t,
        utf-8;q=1.000, UTF-7, unicode-1-1;q=0.210  ,'``, ``str(instance)``
        returns ``'iso-8859-5;q=0, utf-8, UTF-7, unicode-1-1;q=0.21'``.
        """
        return ', '.join((_item_qvalue_pair_to_header_element(pair=tuple_) for tuple_ in self.parsed))

    def _add_instance_and_non_accept_charset_type(self, instance, other, instance_on_the_right=False):
        if not other:
            return self.__class__(header_value=instance.header_value)
        other_header_value = self._python_value_to_header_str(value=other)
        try:
            self.parse(value=other_header_value)
        except ValueError:
            return self.__class__(header_value=instance.header_value)
        new_header_value = other_header_value + ', ' + instance.header_value if instance_on_the_right else instance.header_value + ', ' + other_header_value
        return self.__class__(header_value=new_header_value)

    def _old_match(self, mask, offer):
        """
        Return whether charset offer matches header item (charset or ``*``).

        .. warning::

           This is maintained for backward compatibility, and will be
           deprecated in the future.

        This method was WebOb's old criterion for deciding whether a charset
        matches a header item (charset or ``*``), used in

        - :meth:`AcceptCharsetValidHeader.__contains__`
        - :meth:`AcceptCharsetValidHeader.best_match`
        - :meth:`AcceptCharsetValidHeader.quality`

        It does not conform to :rfc:`RFC 7231, section 5.3.3
        <7231#section-5.3.3>` in that it does not interpret ``*`` values in the
        header correctly: ``*`` should only match charsets not mentioned
        elsewhere in the header.
        """
        return mask == '*' or offer.lower() == mask.lower()

    def acceptable_offers(self, offers):
        """
        Return the offers that are acceptable according to the header.

        The offers are returned in descending order of preference, where
        preference is indicated by the qvalue of the charset or ``*`` in the
        header matching the offer.

        This uses the matching rules described in :rfc:`RFC 7231, section 5.3.3
        <7231#section-5.3.3>`.

        :param offers: ``iterable`` of ``str`` charsets
        :return: A list of tuples of the form (charset, qvalue), in descending
                 order of qvalue. Where two offers have the same qvalue, they
                 are returned in the same order as their order in `offers`.
        """
        lowercased_parsed = [(charset.lower(), qvalue) for charset, qvalue in self.parsed]
        lowercased_offers = [offer.lower() for offer in offers]
        not_acceptable_charsets = set()
        acceptable_charsets = dict()
        asterisk_qvalue = None
        for charset, qvalue in lowercased_parsed:
            if charset == '*':
                if asterisk_qvalue is None:
                    asterisk_qvalue = qvalue
            elif charset not in acceptable_charsets and charset not in not_acceptable_charsets:
                if qvalue == 0.0:
                    not_acceptable_charsets.add(charset)
                else:
                    acceptable_charsets[charset] = qvalue
        acceptable_charsets = list(acceptable_charsets.items())
        acceptable_charsets.sort(key=lambda tuple_: tuple_[1], reverse=True)
        filtered_offers = []
        for index, offer in enumerate(lowercased_offers):
            if any((offer == charset for charset in not_acceptable_charsets)):
                continue
            matched_charset_qvalue = None
            for charset, qvalue in acceptable_charsets:
                if offer == charset:
                    matched_charset_qvalue = qvalue
                    break
            else:
                if asterisk_qvalue:
                    matched_charset_qvalue = asterisk_qvalue
            if matched_charset_qvalue is not None:
                filtered_offers.append((offers[index], matched_charset_qvalue, index))
        filtered_offers.sort(key=lambda tuple_: tuple_[2])
        filtered_offers.sort(key=lambda tuple_: tuple_[1], reverse=True)
        return [(item[0], item[1]) for item in filtered_offers]

    def best_match(self, offers, default_match=None):
        """
        Return the best match from the sequence of charset `offers`.

        .. warning::

           This is currently maintained for backward compatibility, and will be
           deprecated in the future.

           :meth:`AcceptCharsetValidHeader.best_match`  has many issues, and
           does not conform to :rfc:`RFC 7231 <7231>`.

        Each charset in `offers` is checked against each non-``q=0`` item
        (charset or ``*``) in the header. If the two are a match according to
        WebOb's old criterion for a match, the quality value of the match is
        the qvalue of the item from the header multiplied by the server quality
        value of the offer (if the server quality value is not supplied, it is
        1).

        The offer in the match with the highest quality value is the best
        match. If there is more than one match with the highest qvalue, the one
        that shows up first in `offers` is the best match.

        :param offers: (iterable)

                       | Each item in the iterable may be a ``str`` charset, or
                         a (charset, server quality value) ``tuple`` or
                         ``list``.  (The two may be mixed in the iterable.)

        :param default_match: (optional, any type) the value to be returned if
                              there is no match

        :return: (``str``, or the type of `default_match`)

                 | The offer that is the best match. If there is no match, the
                   value of `default_match` is returned.

        The algorithm behind this method was written for the ``Accept`` header
        rather than the ``Accept-Charset`` header. It uses the old criterion of
        a match in :meth:`AcceptCharsetValidHeader._old_match`, which does not
        conform to :rfc:`RFC 7231, section 5.3.3 <7231#section-5.3.3>`, in that
        it does not interpret ``*`` values in the header correctly: ``*``
        should only match charsets not mentioned elsewhere in the header::

            >>> AcceptCharsetValidHeader('utf-8;q=0, *').best_match(['utf-8'])
            'utf-8'
        """
        warnings.warn('The behavior of AcceptCharsetValidHeader.best_match is currently being maintained for backward compatibility, but it will be deprecated in the future, as it does not conform to the RFC.', DeprecationWarning)
        best_quality = -1
        best_offer = default_match
        matched_by = '*/*'
        for offer in offers:
            if isinstance(offer, (tuple, list)):
                offer, server_quality = offer
            else:
                server_quality = 1
            for mask, quality in self._parsed_nonzero:
                possible_quality = server_quality * quality
                if possible_quality < best_quality:
                    continue
                elif possible_quality == best_quality:
                    if matched_by.count('*') <= mask.count('*'):
                        continue
                if self._old_match(mask, offer):
                    best_quality = possible_quality
                    best_offer = offer
                    matched_by = mask
        return best_offer

    def quality(self, offer):
        """
        Return quality value of given offer, or ``None`` if there is no match.

        .. warning::

           This is currently maintained for backward compatibility, and will be
           deprecated in the future.

        :param offer: (``str``) charset offer
        :return: (``float`` or ``None``)

                 | The quality value from the charset that matches the `offer`,
                   or ``None`` if there is no match.

        This uses the old criterion of a match in
        :meth:`AcceptCharsetValidHeader._old_match`, which does not conform to
        :rfc:`RFC 7231, section 5.3.3 <7231#section-5.3.3>`, in that it does
        not interpret ``*`` values in the header correctly: ``*`` should only
        match charsets not mentioned elsewhere in the header::

            >>> AcceptCharsetValidHeader('utf-8;q=0, *').quality('utf-8')
            1.0
            >>> AcceptCharsetValidHeader('utf-8;q=0.9, *').quality('utf-8')
            1.0
        """
        warnings.warn('The behavior of AcceptCharsetValidHeader.quality is currently being maintained for backward compatibility, but it will be deprecated in the future, as it does not conform to the RFC.', DeprecationWarning)
        bestq = 0
        for mask, q in self.parsed:
            if self._old_match(mask, offer):
                bestq = max(bestq, q)
        return bestq or None