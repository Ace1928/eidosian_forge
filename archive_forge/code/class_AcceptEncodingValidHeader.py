from collections import namedtuple
import re
import textwrap
import warnings
class AcceptEncodingValidHeader(AcceptEncoding):
    """
    Represent a valid ``Accept-Encoding`` header.

    A valid header is one that conforms to :rfc:`RFC 7231, section 5.3.4
    <7231#section-5.3.4>`.

    This object should not be modified. To add to the header, we can use the
    addition operators (``+`` and ``+=``), which return a new object (see the
    docstring for :meth:`AcceptEncodingValidHeader.__add__`).
    """

    @property
    def header_value(self):
        """(``str`` or ``None``) The header value."""
        return self._header_value

    @property
    def parsed(self):
        """
        (``list`` or ``None``) Parsed form of the header.

        A list of (*codings*, *qvalue*) tuples, where

        *codings* (``str``) is a content-coding, the string "``identity``", or
        "``*``"; and

        *qvalue* (``float``) is the quality value of the codings.
        """
        return self._parsed

    def __init__(self, header_value):
        """
        Create an :class:`AcceptEncodingValidHeader` instance.

        :param header_value: (``str``) header value.
        :raises ValueError: if `header_value` is an invalid value for an
                            ``Accept-Encoding`` header.
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
        * a ``dict``, with content-coding, ``identity`` or ``*`` ``str``'s as
          keys, and qvalue ``float``'s as values
        * a ``tuple`` or ``list``, where each item is either a header element
          ``str``, or a (content-coding/``identity``/``*``, qvalue) ``tuple``
          or ``list``
        * an :class:`AcceptEncodingValidHeader`,
          :class:`AcceptEncodingNoHeader`, or
          :class:`AcceptEncodingInvalidHeader` instance
        * object of any other type that returns a value for ``__str__``

        If `other` is a valid header value or another
        :class:`AcceptEncodingValidHeader` instance, and the header value it
        represents is not ``''``, then the two header values are joined with
        ``', '``, and a new :class:`AcceptEncodingValidHeader` instance with
        the new header value is returned.

        If `other` is a valid header value or another
        :class:`AcceptEncodingValidHeader` instance representing a header value
        of ``''``; or if it is ``None`` or an :class:`AcceptEncodingNoHeader`
        instance; or if it is an invalid header value, or an
        :class:`AcceptEncodingInvalidHeader` instance, then a new
        :class:`AcceptEncodingValidHeader` instance with the same header value
        as ``self`` is returned.
        """
        if isinstance(other, AcceptEncodingValidHeader):
            if other.header_value == '':
                return self.__class__(header_value=self.header_value)
            else:
                return create_accept_encoding_header(header_value=self.header_value + ', ' + other.header_value)
        if isinstance(other, (AcceptEncodingNoHeader, AcceptEncodingInvalidHeader)):
            return self.__class__(header_value=self.header_value)
        return self._add_instance_and_non_accept_encoding_type(instance=self, other=other)

    def __bool__(self):
        """
        Return whether ``self`` represents a valid ``Accept-Encoding`` header.

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

           The behavior of :meth:`AcceptEncodingValidHeader.__contains__` is
           currently being maintained for backward compatibility, but it will
           change in the future to better conform to the RFC.

        :param offer: (``str``) a content-coding or ``identity`` offer
        :return: (``bool``) Whether ``offer`` is acceptable according to the
                 header.

        The behavior of this method does not fully conform to :rfc:`7231`.
        It does not correctly interpret ``*``::

            >>> 'gzip' in AcceptEncodingValidHeader('gzip;q=0, *')
            True

        and does not handle the ``identity`` token correctly::

            >>> 'identity' in AcceptEncodingValidHeader('gzip')
            False
        """
        warnings.warn('The behavior of AcceptEncodingValidHeader.__contains__ is currently being maintained for backward compatibility, but it will change in the future to better conform to the RFC.', DeprecationWarning)
        for mask, quality in self._parsed_nonzero:
            if self._old_match(mask, offer):
                return True

    def __iter__(self):
        """
        Return all the ranges with non-0 qvalues, in order of preference.

        .. warning::

           The behavior of this method is currently maintained for backward
           compatibility, but will change in the future.

        :return: iterator of all the (content-coding/``identity``/``*``) items
                 in the header with non-0 qvalues, in descending order of
                 qvalue. If two items have the same qvalue, they are returned
                 in the order of their positions in the header, from left to
                 right.

        Please note that this is a simple filter for the items in the header
        with non-0 qvalues, and is not necessarily the same as what the client
        prefers, e.g. ``'gzip;q=0, *'`` means 'everything but gzip', but
        ``list(instance)`` would return only ``['*']``.
        """
        warnings.warn('The behavior of AcceptEncodingLanguageValidHeader.__iter__ is currently maintained for backward compatibility, but will change in the future.', DeprecationWarning)
        for m, q in sorted(self._parsed_nonzero, key=lambda i: i[1], reverse=True):
            yield m

    def __radd__(self, other):
        """
        Add to header, creating a new header object.

        See the docstring for :meth:`AcceptEncodingValidHeader.__add__`.
        """
        return self._add_instance_and_non_accept_encoding_type(instance=self, other=other, instance_on_the_right=True)

    def __repr__(self):
        return '<{} ({!r})>'.format(self.__class__.__name__, str(self))

    def __str__(self):
        """
        Return a tidied up version of the header value.

        e.g. If the ``header_value`` is ``",\\t, a ;\\t q=0.20 , b ,',"``,
        ``str(instance)`` returns ``"a;q=0.2, b, '"``.
        """
        return ', '.join((_item_qvalue_pair_to_header_element(pair=tuple_) for tuple_ in self.parsed))

    def _add_instance_and_non_accept_encoding_type(self, instance, other, instance_on_the_right=False):
        if not other:
            return self.__class__(header_value=instance.header_value)
        other_header_value = self._python_value_to_header_str(value=other)
        if other_header_value == '':
            return self.__class__(header_value=instance.header_value)
        try:
            self.parse(value=other_header_value)
        except ValueError:
            return self.__class__(header_value=instance.header_value)
        new_header_value = other_header_value + ', ' + instance.header_value if instance_on_the_right else instance.header_value + ', ' + other_header_value
        return self.__class__(header_value=new_header_value)

    def _old_match(self, mask, offer):
        """
        Return whether content-coding offer matches codings header item.

        .. warning::

           This is maintained for backward compatibility, and will be
           deprecated in the future.

        This method was WebOb's old criterion for deciding whether a
        content-coding offer matches a header item (content-coding,
        ``identity`` or ``*``), used in

        - :meth:`AcceptCharsetValidHeader.__contains__`
        - :meth:`AcceptCharsetValidHeader.best_match`
        - :meth:`AcceptCharsetValidHeader.quality`

        It does not conform to :rfc:`RFC 7231, section 5.3.4
        <7231#section-5.3.4>` in that it does not interpret ``*`` values in the
        header correctly: ``*`` should only match content-codings not mentioned
        elsewhere in the header.
        """
        return mask == '*' or offer.lower() == mask.lower()

    def acceptable_offers(self, offers):
        """
        Return the offers that are acceptable according to the header.

        The offers are returned in descending order of preference, where
        preference is indicated by the qvalue of the item (content-coding,
        "identity" or "*") in the header that matches the offer.

        This uses the matching rules described in :rfc:`RFC 7231, section 5.3.4
        <7231#section-5.3.4>`.

        :param offers: ``iterable`` of ``str``s, where each ``str`` is a
                       content-coding or the string ``identity`` (the token
                       used to represent "no encoding")
        :return: A list of tuples of the form (content-coding or "identity",
                 qvalue), in descending order of qvalue. Where two offers have
                 the same qvalue, they are returned in the same order as their
                 order in `offers`.

        Use the string ``'identity'`` (without the quotes) in `offers` to
        indicate an offer with no content-coding. From the RFC: 'If the
        representation has no content-coding, then it is acceptable by default
        unless specifically excluded by the Accept-Encoding field stating
        either "identity;q=0" or "\\*;q=0" without a more specific entry for
        "identity".' The RFC does not specify the qvalue that should be
        assigned to the representation/offer with no content-coding; this
        implementation assigns it a qvalue of 1.0.
        """
        lowercased_parsed = [(codings.lower(), qvalue) for codings, qvalue in self.parsed]
        lowercased_offers = [offer.lower() for offer in offers]
        not_acceptable_codingss = set()
        acceptable_codingss = dict()
        asterisk_qvalue = None
        for codings, qvalue in lowercased_parsed:
            if codings == '*':
                if asterisk_qvalue is None:
                    asterisk_qvalue = qvalue
            elif codings not in acceptable_codingss and codings not in not_acceptable_codingss:
                if qvalue == 0.0:
                    not_acceptable_codingss.add(codings)
                else:
                    acceptable_codingss[codings] = qvalue
        acceptable_codingss = list(acceptable_codingss.items())
        acceptable_codingss.sort(key=lambda tuple_: tuple_[1], reverse=True)
        filtered_offers = []
        for index, offer in enumerate(lowercased_offers):
            if any((offer == codings for codings in not_acceptable_codingss)):
                continue
            matched_codings_qvalue = None
            for codings, qvalue in acceptable_codingss:
                if offer == codings:
                    matched_codings_qvalue = qvalue
                    break
            else:
                if asterisk_qvalue:
                    matched_codings_qvalue = asterisk_qvalue
                elif asterisk_qvalue != 0.0 and offer == 'identity':
                    matched_codings_qvalue = 1.0
            if matched_codings_qvalue is not None:
                filtered_offers.append((offers[index], matched_codings_qvalue, index))
        filtered_offers.sort(key=lambda tuple_: tuple_[2])
        filtered_offers.sort(key=lambda tuple_: tuple_[1], reverse=True)
        return [(item[0], item[1]) for item in filtered_offers]

    def best_match(self, offers, default_match=None):
        """
        Return the best match from the sequence of `offers`.

        .. warning::

           This is currently maintained for backward compatibility, and will be
           deprecated in the future.

           :meth:`AcceptEncodingValidHeader.best_match` uses its own algorithm
           (one not specified in :rfc:`RFC 7231 <7231>`) to determine what is a
           best match. The algorithm has many issues, and does not conform to
           the RFC.

        Each offer in `offers` is checked against each non-``q=0`` item
        (content-coding/``identity``/``*``) in the header. If the two are a
        match according to WebOb's old criterion for a match, the quality value
        of the match is the qvalue of the item from the header multiplied by
        the server quality value of the offer (if the server quality value is
        not supplied, it is 1).

        The offer in the match with the highest quality value is the best
        match. If there is more than one match with the highest qvalue, the one
        that shows up first in `offers` is the best match.

        :param offers: (iterable)

                       | Each item in the iterable may be a ``str`` *codings*,
                         or a (*codings*, server quality value) ``tuple`` or
                         ``list``, where *codings* is either a content-coding,
                         or the string ``identity`` (which represents *no
                         encoding*). ``str`` and ``tuple``/``list`` elements
                         may be mixed within the iterable.

        :param default_match: (optional, any type) the value to be returned if
                              there is no match

        :return: (``str``, or the type of `default_match`)

                 | The offer that is the best match. If there is no match, the
                   value of `default_match` is returned.

        This method does not conform to :rfc:`RFC 7231, section 5.3.4
        <7231#section-5.3.4>`, in that it does not correctly interpret ``*``::

            >>> AcceptEncodingValidHeader('gzip;q=0, *').best_match(['gzip'])
            'gzip'

        and does not handle the ``identity`` token correctly::

            >>> instance = AcceptEncodingValidHeader('gzip')
            >>> instance.best_match(['identity']) is None
            True
        """
        warnings.warn('The behavior of AcceptEncodingValidHeader.best_match is currently being maintained for backward compatibility, but it will be deprecated in the future, as it does not conform to the RFC.', DeprecationWarning)
        best_quality = -1
        best_offer = default_match
        matched_by = '*/*'
        for offer in offers:
            if isinstance(offer, (tuple, list)):
                offer, server_quality = offer
            else:
                server_quality = 1
            for item in self._parsed_nonzero:
                mask = item[0]
                quality = item[1]
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

        :param offer: (``str``) A content-coding, or ``identity``.
        :return: (``float`` or ``None``)

                 | The quality value from the header item
                   (content-coding/``identity``/``*``) that matches the
                   `offer`, or ``None`` if there is no match.

        The behavior of this method does not conform to :rfc:`RFC 7231, section
        5.3.4<7231#section-5.3.4>`, in that it does not correctly interpret
        ``*``::

            >>> AcceptEncodingValidHeader('gzip;q=0, *').quality('gzip')
            1.0

        and does not handle the ``identity`` token correctly::

            >>> AcceptEncodingValidHeader('gzip').quality('identity') is None
            True
        """
        warnings.warn('The behavior of AcceptEncodingValidHeader.quality is currently being maintained for backward compatibility, but it will be deprecated in the future, as it does not conform to the RFC.', DeprecationWarning)
        bestq = 0
        for mask, q in self.parsed:
            if self._old_match(mask, offer):
                bestq = max(bestq, q)
        return bestq or None