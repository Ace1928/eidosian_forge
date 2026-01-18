from collections import namedtuple
import re
import textwrap
import warnings
class AcceptValidHeader(Accept):
    """
    Represent a valid ``Accept`` header.

    A valid header is one that conforms to :rfc:`RFC 7231, section 5.3.2
    <7231#section-5.3.2>`.

    This object should not be modified. To add to the header, we can use the
    addition operators (``+`` and ``+=``), which return a new object (see the
    docstring for :meth:`AcceptValidHeader.__add__`).
    """

    @property
    def header_value(self):
        """(``str`` or ``None``) The header value."""
        return self._header_value

    @property
    def parsed(self):
        """
        (``list`` or ``None``) Parsed form of the header.

        A list of (*media_range*, *qvalue*, *media_type_params*,
        *extension_params*) tuples, where

        *media_range* is the media range, including any media type parameters.
        The media range is returned in a canonicalised form (except the case of
        the characters are unchanged): unnecessary spaces around the semicolons
        before media type parameters are removed; the parameter values are
        returned in a form where only the '``\\``' and '``"``' characters are
        escaped, and the values are quoted with double quotes only if they need
        to be quoted.

        *qvalue* is the quality value of the media range.

        *media_type_params* is the media type parameters, as a list of
        (parameter name, value) tuples.

        *extension_params* is the extension parameters, as a list where each
        item is either a parameter string or a (parameter name, value) tuple.
        """
        return self._parsed

    def __init__(self, header_value):
        """
        Create an :class:`AcceptValidHeader` instance.

        :param header_value: (``str``) header value.
        :raises ValueError: if `header_value` is an invalid value for an
                            ``Accept`` header.
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

        If `other` is a valid header value or another
        :class:`AcceptValidHeader` instance, and the header value it represents
        is not `''`, then the two header values are joined with ``', '``, and a
        new :class:`AcceptValidHeader` instance with the new header value is
        returned.

        If `other` is a valid header value or another
        :class:`AcceptValidHeader` instance representing a header value of
        `''`; or if it is ``None`` or an :class:`AcceptNoHeader` instance; or
        if it is an invalid header value, or an :class:`AcceptInvalidHeader`
        instance, then a new :class:`AcceptValidHeader` instance with the same
        header value as ``self`` is returned.
        """
        if isinstance(other, AcceptValidHeader):
            if other.header_value == '':
                return self.__class__(header_value=self.header_value)
            else:
                return create_accept_header(header_value=self.header_value + ', ' + other.header_value)
        if isinstance(other, (AcceptNoHeader, AcceptInvalidHeader)):
            return self.__class__(header_value=self.header_value)
        return self._add_instance_and_non_accept_type(instance=self, other=other)

    def __bool__(self):
        """
        Return whether ``self`` represents a valid ``Accept`` header.

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

           The behavior of :meth:`AcceptValidHeader.__contains__` is currently
           being maintained for backward compatibility, but it will change in
           the future to better conform to the RFC.

        :param offer: (``str``) media type offer
        :return: (``bool``) Whether ``offer`` is acceptable according to the
                 header.

        This uses the old criterion of a match in
        :meth:`AcceptValidHeader._old_match`, which is not as specified in
        :rfc:`RFC 7231, section 5.3.2 <7231#section-5.3.2>`. It does not
        correctly take into account media type parameters:

            >>> 'text/html;p=1' in AcceptValidHeader('text/html')
            False

        or media ranges with ``q=0`` in the header::

            >>> 'text/html' in AcceptValidHeader('text/*, text/html;q=0')
            True
            >>> 'text/html' in AcceptValidHeader('text/html;q=0, */*')
            True

        (See the docstring for :meth:`AcceptValidHeader._old_match` for other
        problems with the old criterion for matching.)
        """
        warnings.warn('The behavior of AcceptValidHeader.__contains__ is currently being maintained for backward compatibility, but it will change in the future to better conform to the RFC.', DeprecationWarning)
        for media_range, quality, media_type_params, extension_params in self._parsed_nonzero:
            if self._old_match(media_range, offer):
                return True
        return False

    def __iter__(self):
        """
        Return all the ranges with non-0 qvalues, in order of preference.

        .. warning::

           The behavior of this method is currently maintained for backward
           compatibility, but will change in the future.

        :return: iterator of all the media ranges in the header with non-0
                 qvalues, in descending order of qvalue. If two ranges have the
                 same qvalue, they are returned in the order of their positions
                 in the header, from left to right.

        Please note that this is a simple filter for the ranges in the header
        with non-0 qvalues, and is not necessarily the same as what the client
        prefers, e.g. ``'audio/basic;q=0, */*'`` means 'everything but
        audio/basic', but ``list(instance)`` would return only ``['*/*']``.
        """
        warnings.warn('The behavior of AcceptLanguageValidHeader.__iter__ is currently maintained for backward compatibility, but will change in the future.', DeprecationWarning)
        for media_range, qvalue, media_type_params, extension_params in sorted(self._parsed_nonzero, key=lambda i: i[1], reverse=True):
            yield media_range

    def __radd__(self, other):
        """
        Add to header, creating a new header object.

        See the docstring for :meth:`AcceptValidHeader.__add__`.
        """
        return self._add_instance_and_non_accept_type(instance=self, other=other, instance_on_the_right=True)

    def __repr__(self):
        return '<{} ({!r})>'.format(self.__class__.__name__, str(self))

    def __str__(self):
        """
        Return a tidied up version of the header value.

        e.g. If ``self.header_value`` is ``r',,text/html ; p1="\\"\\1\\"" ;
        q=0.50; e1=1 ;e2  ,  text/plain ,'``, ``str(instance)`` returns
        ``r'text/html;p1="\\"1\\"";q=0.5;e1=1;e2, text/plain'``.
        """
        return ', '.join((self._iterable_to_header_element(iterable=(tuple_[0], tuple_[1], self._form_extension_params_segment(extension_params=tuple_[3]))) for tuple_ in self.parsed))

    def _add_instance_and_non_accept_type(self, instance, other, instance_on_the_right=False):
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
        Check if the offer is covered by the mask

        ``offer`` may contain wildcards to facilitate checking if a ``mask``
        would match a 'permissive' offer.

        Wildcard matching forces the match to take place against the type or
        subtype of the mask and offer (depending on where the wildcard matches)

        .. warning::

           This is maintained for backward compatibility, and will be
           deprecated in the future.

        This method was WebOb's old criterion for deciding whether a media type
        matches a media range, used in

        - :meth:`AcceptValidHeader.__contains__`
        - :meth:`AcceptValidHeader.best_match`
        - :meth:`AcceptValidHeader.quality`

        It allows offers of *, */*, type/*, */subtype and types with no
        subtypes, which are not media types as specified in :rfc:`RFC 7231,
        section 5.3.2 <7231#section-5.3.2>`. This is also undocumented in any
        of the public APIs that uses this method.
        """
        if mask.lower() == offer.lower() or '*/*' in (mask, offer) or '*' == offer:
            return True
        try:
            mask_type, mask_subtype = [x.lower() for x in mask.split('/')]
        except ValueError:
            mask_type = mask
            mask_subtype = '*'
        try:
            offer_type, offer_subtype = [x.lower() for x in offer.split('/')]
        except ValueError:
            offer_type = offer
            offer_subtype = '*'
        if mask_subtype == '*':
            if offer_type == '*':
                return True
            else:
                return mask_type.lower() == offer_type.lower()
        if mask_type == '*':
            if offer_subtype == '*':
                return True
            else:
                return mask_subtype.lower() == offer_subtype.lower()
        if offer_subtype == '*':
            return mask_type.lower() == offer_type.lower()
        if offer_type == '*':
            return mask_subtype.lower() == offer_subtype.lower()
        return offer.lower() == mask.lower()

    def accept_html(self):
        """
        Return ``True`` if any HTML-like type is accepted.

        The HTML-like types are 'text/html', 'application/xhtml+xml',
        'application/xml' and 'text/xml'.
        """
        return bool(self.acceptable_offers(offers=['text/html', 'application/xhtml+xml', 'application/xml', 'text/xml']))
    accepts_html = property(fget=accept_html, doc=accept_html.__doc__)

    def acceptable_offers(self, offers):
        """
        Return the offers that are acceptable according to the header.

        The offers are returned in descending order of preference, where
        preference is indicated by the qvalue of the media range in the header
        that best matches the offer.

        This uses the matching rules described in :rfc:`RFC 7231, section 5.3.2
        <7231#section-5.3.2>`.

        Any offers that cannot be parsed via
        :meth:`.Accept.parse_offer` will be ignored.

        :param offers: ``iterable`` of ``str`` media types (media types can
                       include media type parameters) or pre-parsed instances
                       of :class:`.AcceptOffer`.
        :return: A list of tuples of the form (media type, qvalue), in
                 descending order of qvalue. Where two offers have the same
                 qvalue, they are returned in the same order as their order in
                 `offers`.
        """
        parsed = self.parsed
        lowercased_ranges = [(media_range.partition(';')[0].lower(), qvalue, tuple(((name.lower(), value) for name, value in media_type_params))) for media_range, qvalue, media_type_params, __ in parsed]
        lowercased_offers_parsed = self._parse_and_normalize_offers(offers)
        acceptable_offers_n_quality_factors = {}
        for offer_index, parsed_offer in lowercased_offers_parsed:
            offer = offers[offer_index]
            offer_type, offer_subtype, offer_media_type_params = parsed_offer
            for range_type_subtype, range_qvalue, range_media_type_params in lowercased_ranges:
                range_type, range_subtype = range_type_subtype.split('/', 1)
                if offer_type == range_type and offer_subtype == range_subtype:
                    if range_media_type_params == ():
                        specificity = 3
                    elif offer_media_type_params == range_media_type_params:
                        specificity = 4
                    else:
                        continue
                elif range_subtype == '*' and offer_type == range_type:
                    specificity = 2
                elif range_type_subtype == '*/*':
                    specificity = 1
                else:
                    continue
                try:
                    if specificity <= acceptable_offers_n_quality_factors[offer][2]:
                        continue
                except KeyError:
                    pass
                acceptable_offers_n_quality_factors[offer] = (range_qvalue, offer_index, specificity)
        acceptable_offers_n_quality_factors = [(key, value[0], value[1]) for key, value in acceptable_offers_n_quality_factors.items() if value[0]]
        acceptable_offers_n_quality_factors.sort(key=lambda tuple_: tuple_[2])
        acceptable_offers_n_quality_factors.sort(key=lambda tuple_: tuple_[1], reverse=True)
        acceptable_offers_n_quality_factors = [(item[0], item[1]) for item in acceptable_offers_n_quality_factors]
        return acceptable_offers_n_quality_factors

    def best_match(self, offers, default_match=None):
        """
        Return the best match from the sequence of media type `offers`.

        .. warning::

           This is currently maintained for backward compatibility, and will be
           deprecated in the future.

           :meth:`AcceptValidHeader.best_match` uses its own algorithm (one not
           specified in :rfc:`RFC 7231 <7231>`) to determine what is a best
           match. The algorithm has many issues, and does not conform to
           :rfc:`RFC 7231 <7231>`.

        Each media type in `offers` is checked against each non-``q=0`` range
        in the header. If the two are a match according to WebOb's old
        criterion for a match, the quality value of the match is the qvalue of
        the media range from the header multiplied by the server quality value
        of the offer (if the server quality value is not supplied, it is 1).

        The offer in the match with the highest quality value is the best
        match. If there is more than one match with the highest qvalue, the
        match where the media range has a lower number of '*'s is the best
        match. If the two have the same number of '*'s, the one that shows up
        first in `offers` is the best match.

        :param offers: (iterable)

                       | Each item in the iterable may be a ``str`` media type,
                         or a (media type, server quality value) ``tuple`` or
                         ``list``. (The two may be mixed in the iterable.)

        :param default_match: (optional, any type) the value to be returned if
                              there is no match

        :return: (``str``, or the type of `default_match`)

                 | The offer that is the best match. If there is no match, the
                   value of `default_match` is returned.

        This uses the old criterion of a match in
        :meth:`AcceptValidHeader._old_match`, which is not as specified in
        :rfc:`RFC 7231, section 5.3.2 <7231#section-5.3.2>`. It does not
        correctly take into account media type parameters:

            >>> instance = AcceptValidHeader('text/html')
            >>> instance.best_match(offers=['text/html;p=1']) is None
            True

        or media ranges with ``q=0`` in the header::

            >>> instance = AcceptValidHeader('text/*, text/html;q=0')
            >>> instance.best_match(offers=['text/html'])
            'text/html'

            >>> instance = AcceptValidHeader('text/html;q=0, */*')
            >>> instance.best_match(offers=['text/html'])
            'text/html'

        (See the docstring for :meth:`AcceptValidHeader._old_match` for other
        problems with the old criterion for matching.)

        Another issue is that this method considers the best matching range for
        an offer to be the matching range with the highest quality value,
        (where quality values are tied, the most specific media range is
        chosen); whereas :rfc:`RFC 7231, section 5.3.2 <7231#section-5.3.2>`
        specifies that we should consider the best matching range for a media
        type offer to be the most specific matching range.::

            >>> instance = AcceptValidHeader('text/html;q=0.5, text/*')
            >>> instance.best_match(offers=['text/html', 'text/plain'])
            'text/html'
        """
        warnings.warn('The behavior of AcceptValidHeader.best_match is currently being maintained for backward compatibility, but it will be deprecated in the future, as it does not conform to the RFC.', DeprecationWarning)
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

        :param offer: (``str``) media type offer
        :return: (``float`` or ``None``)

                 | The highest quality value from the media range(s) that match
                   the `offer`, or ``None`` if there is no match.

        This uses the old criterion of a match in
        :meth:`AcceptValidHeader._old_match`, which is not as specified in
        :rfc:`RFC 7231, section 5.3.2 <7231#section-5.3.2>`. It does not
        correctly take into account media type parameters:

            >>> instance = AcceptValidHeader('text/html')
            >>> instance.quality('text/html;p=1') is None
            True

        or media ranges with ``q=0`` in the header::

            >>> instance = AcceptValidHeader('text/*, text/html;q=0')
            >>> instance.quality('text/html')
            1.0
            >>> AcceptValidHeader('text/html;q=0, */*').quality('text/html')
            1.0

        (See the docstring for :meth:`AcceptValidHeader._old_match` for other
        problems with the old criterion for matching.)

        Another issue is that this method considers the best matching range for
        an offer to be the matching range with the highest quality value,
        whereas :rfc:`RFC 7231, section 5.3.2 <7231#section-5.3.2>` specifies
        that we should consider the best matching range for a media type offer
        to be the most specific matching range.::

            >>> instance = AcceptValidHeader('text/html;q=0.5, text/*')
            >>> instance.quality('text/html')
            1.0
        """
        warnings.warn('The behavior of AcceptValidHeader.quality is currently being maintained for backward compatibility, but it will be deprecated in the future, as it does not conform to the RFC.', DeprecationWarning)
        bestq = 0
        for item in self.parsed:
            media_range = item[0]
            qvalue = item[1]
            if self._old_match(media_range, offer):
                bestq = max(bestq, qvalue)
        return bestq or None