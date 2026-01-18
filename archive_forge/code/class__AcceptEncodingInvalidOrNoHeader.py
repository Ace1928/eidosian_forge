from collections import namedtuple
import re
import textwrap
import warnings
class _AcceptEncodingInvalidOrNoHeader(AcceptEncoding):
    """
    Represent when an ``Accept-Encoding`` header is invalid or not in request.

    This is the base class for the behaviour that
    :class:`.AcceptEncodingInvalidHeader` and :class:`.AcceptEncodingNoHeader`
    have in common.

    :rfc:`7231` does not provide any guidance on what should happen if the
    ``AcceptEncoding`` header has an invalid value. This implementation
    disregards the header when the header is invalid, so
    :class:`.AcceptEncodingInvalidHeader` and :class:`.AcceptEncodingNoHeader`
    have much behaviour in common.
    """

    def __bool__(self):
        """
        Return whether ``self`` represents a valid ``Accept-Encoding`` header.

        Return ``True`` if ``self`` represents a valid header, and ``False`` if
        it represents an invalid header, or the header not being in the
        request.

        For this class, it always returns ``False``.
        """
        return False
    __nonzero__ = __bool__

    def __contains__(self, offer):
        """
        Return ``bool`` indicating whether `offer` is acceptable.

        .. warning::

           The behavior of ``.__contains__`` for the ``Accept-Encoding``
           classes is currently being maintained for backward compatibility,
           but it will change in the future to better conform to the RFC.

        :param offer: (``str``) a content-coding or ``identity`` offer
        :return: (``bool``) Whether ``offer`` is acceptable according to the
                 header.

        For this class, either there is no ``Accept-Encoding`` header in the
        request, or the header is invalid, so any content-coding is acceptable,
        and this always returns ``True``.
        """
        warnings.warn('The behavior of .__contains__ for the Accept-Encoding classes is currently being maintained for backward compatibility, but it will change in the future to better conform to the RFC.', DeprecationWarning)
        return True

    def __iter__(self):
        """
        Return all the header items with non-0 qvalues, in order of preference.

        .. warning::

           The behavior of this method is currently maintained for backward
           compatibility, but will change in the future.

        :return: iterator of all the (content-coding/``identity``/``*``) items
                 in the header with non-0 qvalues, in descending order of
                 qvalue. If two items have the same qvalue, they are returned
                 in the order of their positions in the header, from left to
                 right.

        When there is no ``Accept-Encoding`` header in the request or the
        header is invalid, there are no items in the header, so this always
        returns an empty iterator.
        """
        warnings.warn('The behavior of AcceptEncodingValidHeader.__iter__ is currently maintained for backward compatibility, but will change in the future.', DeprecationWarning)
        return iter(())

    def acceptable_offers(self, offers):
        """
        Return the offers that are acceptable according to the header.

        :param offers: ``iterable`` of ``str``s, where each ``str`` is a
                       content-coding or the string ``identity`` (the token
                       used to represent "no encoding")
        :return: When the header is invalid, or there is no ``Accept-Encoding``
                 header in the request, all `offers` are considered acceptable,
                 so this method returns a list of (content-coding or
                 "identity", qvalue) tuples where each offer in `offers` is
                 paired with the qvalue of 1.0, in the same order as in
                 `offers`.
        """
        return [(offer, 1.0) for offer in offers]

    def best_match(self, offers, default_match=None):
        """
        Return the best match from the sequence of `offers`.

        This is the ``.best_match()`` method for when the header is invalid or
        not found in the request, corresponding to
        :meth:`AcceptEncodingValidHeader.best_match`.

        .. warning::

           This is currently maintained for backward compatibility, and will be
           deprecated in the future (see the documentation for
           :meth:`AcceptEncodingValidHeader.best_match`).

        When the header is invalid, or there is no `Accept-Encoding` header in
        the request, all `offers` are considered acceptable, so the best match
        is the offer in `offers` with the highest server quality value (if the
        server quality value is not supplied for a media type, it is 1).

        If more than one offer in `offers` have the same highest server quality
        value, then the one that shows up first in `offers` is the best match.

        :param offers: (iterable)

                       | Each item in the iterable may be a ``str`` *codings*,
                         or a (*codings*, server quality value) ``tuple`` or
                         ``list``, where *codings* is either a content-coding,
                         or the string ``identity`` (which represents *no
                         encoding*). ``str`` and ``tuple``/``list`` elements
                         may be mixed within the iterable.

        :param default_match: (optional, any type) the value to be returned if
                              `offers` is empty.

        :return: (``str``, or the type of `default_match`)

                 | The offer that has the highest server quality value. If
                   `offers` is empty, the value of `default_match` is returned.
        """
        warnings.warn('The behavior of .best_match for the Accept-Encoding classes is currently being maintained for backward compatibility, but the method will be deprecated in the future, as its behavior is not specified in (and currently does not conform to) RFC 7231.', DeprecationWarning)
        best_quality = -1
        best_offer = default_match
        for offer in offers:
            if isinstance(offer, (list, tuple)):
                offer, quality = offer
            else:
                quality = 1
            if quality > best_quality:
                best_offer = offer
                best_quality = quality
        return best_offer

    def quality(self, offer):
        """
        Return quality value of given offer, or ``None`` if there is no match.

        This is the ``.quality()`` method for when the header is invalid or not
        found in the request, corresponding to
        :meth:`AcceptEncodingValidHeader.quality`.

        .. warning::

           This is currently maintained for backward compatibility, and will be
           deprecated in the future (see the documentation for
           :meth:`AcceptEncodingValidHeader.quality`).

        :param offer: (``str``) A content-coding, or ``identity``.
        :return: (``float``) ``1.0``.

        When the ``Accept-Encoding`` header is invalid or not in the request,
        all offers are equally acceptable, so 1.0 is always returned.
        """
        warnings.warn('The behavior of .quality for the Accept-Encoding classes is currently being maintained for backward compatibility, but the method will be deprecated in the future, as its behavior does not conform to RFC 7231.', DeprecationWarning)
        return 1.0