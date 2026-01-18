from collections import namedtuple
import re
import textwrap
import warnings
class AcceptLanguageValidHeader(AcceptLanguage):
    """
    Represent a valid ``Accept-Language`` header.

    A valid header is one that conforms to :rfc:`RFC 7231, section 5.3.5
    <7231#section-5.3.5>`.

    We take the reference from the ``language-range`` syntax rule in :rfc:`RFC
    7231, section 5.3.5 <7231#section-5.3.5>` to :rfc:`RFC 4647, section 2.1
    <4647#section-2.1>` to mean that only basic language ranges (and not
    extended language ranges) are expected in the ``Accept-Language`` header.

    This object should not be modified. To add to the header, we can use the
    addition operators (``+`` and ``+=``), which return a new object (see the
    docstring for :meth:`AcceptLanguageValidHeader.__add__`).
    """

    def __init__(self, header_value):
        """
        Create an :class:`AcceptLanguageValidHeader` instance.

        :param header_value: (``str``) header value.
        :raises ValueError: if `header_value` is an invalid value for an
                            ``Accept-Language`` header.
        """
        self._header_value = header_value
        self._parsed = list(self.parse(header_value))
        self._parsed_nonzero = [item for item in self.parsed if item[1]]

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

        A list of (language range, quality value) tuples.
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

        If `other` is a valid header value or another
        :class:`AcceptLanguageValidHeader` instance, the two header values are
        joined with ``', '``, and a new :class:`AcceptLanguageValidHeader`
        instance with the new header value is returned.

        If `other` is ``None``, an :class:`AcceptLanguageNoHeader` instance, an
        invalid header value, or an :class:`AcceptLanguageInvalidHeader`
        instance, a new :class:`AcceptLanguageValidHeader` instance with the
        same header value as ``self`` is returned.
        """
        if isinstance(other, AcceptLanguageValidHeader):
            return create_accept_language_header(header_value=self.header_value + ', ' + other.header_value)
        if isinstance(other, (AcceptLanguageNoHeader, AcceptLanguageInvalidHeader)):
            return self.__class__(header_value=self.header_value)
        return self._add_instance_and_non_accept_language_type(instance=self, other=other)

    def __nonzero__(self):
        """
        Return whether ``self`` represents a valid ``Accept-Language`` header.

        Return ``True`` if ``self`` represents a valid header, and ``False`` if
        it represents an invalid header, or the header not being in the
        request.

        For this class, it always returns ``True``.
        """
        return True
    __bool__ = __nonzero__

    def __contains__(self, offer):
        """
        Return ``bool`` indicating whether `offer` is acceptable.

        .. warning::

           The behavior of :meth:`AcceptLanguageValidHeader.__contains__` is
           currently being maintained for backward compatibility, but it will
           change in the future to better conform to the RFC.

           What is 'acceptable' depends on the needs of your application.
           :rfc:`RFC 7231, section 5.3.5 <7231#section-5.3.5>` suggests three
           matching schemes from :rfc:`RFC 4647 <4647>`, two of which WebOb
           supports with :meth:`AcceptLanguageValidHeader.basic_filtering` and
           :meth:`AcceptLanguageValidHeader.lookup` (we interpret the RFC to
           mean that Extended Filtering cannot apply for the
           ``Accept-Language`` header, as the header only accepts basic
           language ranges.) If these are not suitable for the needs of your
           application, you may need to write your own matching using
           :attr:`AcceptLanguageValidHeader.parsed`.

        :param offer: (``str``) language tag offer
        :return: (``bool``) Whether ``offer`` is acceptable according to the
                 header.

        This uses the old criterion of a match in
        :meth:`AcceptLanguageValidHeader._old_match`, which does not conform to
        :rfc:`RFC 7231, section 5.3.5 <7231#section-5.3.5>` or any of the
        matching schemes suggested there. It also does not properly take into
        account ranges with ``q=0`` in the header::

            >>> 'en-gb' in AcceptLanguageValidHeader('en, en-gb;q=0')
            True
            >>> 'en' in AcceptLanguageValidHeader('en;q=0, *')
            True

        (See the docstring for :meth:`AcceptLanguageValidHeader._old_match` for
        other problems with the old criterion for a match.)
        """
        warnings.warn('The behavior of AcceptLanguageValidHeader.__contains__ is currently being maintained for backward compatibility, but it will change in the future to better conform to the RFC.', DeprecationWarning)
        for mask, quality in self._parsed_nonzero:
            if self._old_match(mask, offer):
                return True
        return False

    def __iter__(self):
        """
        Return all the ranges with non-0 qvalues, in order of preference.

        .. warning::

           The behavior of this method is currently maintained for backward
           compatibility, but will change in the future.

        :return: iterator of all the language ranges in the header with non-0
                 qvalues, in descending order of qvalue. If two ranges have the
                 same qvalue, they are returned in the order of their positions
                 in the header, from left to right.

        Please note that this is a simple filter for the ranges in the header
        with non-0 qvalues, and is not necessarily the same as what the client
        prefers, e.g. ``'en-gb;q=0, *'`` means 'everything but British
        English', but ``list(instance)`` would return only ``['*']``.
        """
        warnings.warn('The behavior of AcceptLanguageValidHeader.__iter__ is currently maintained for backward compatibility, but will change in the future.', DeprecationWarning)
        for m, q in sorted(self._parsed_nonzero, key=lambda i: i[1], reverse=True):
            yield m

    def __radd__(self, other):
        """
        Add to header, creating a new header object.

        See the docstring for :meth:`AcceptLanguageValidHeader.__add__`.
        """
        return self._add_instance_and_non_accept_language_type(instance=self, other=other, instance_on_the_right=True)

    def __repr__(self):
        return '<{} ({!r})>'.format(self.__class__.__name__, str(self))

    def __str__(self):
        """
        Return a tidied up version of the header value.

        e.g. If the ``header_value`` is ``', \\t,de;q=0.000 \\t, es;q=1.000, zh,
        jp;q=0.210  ,'``, ``str(instance)`` returns ``'de;q=0, es, zh,
        jp;q=0.21'``.
        """
        return ', '.join((_item_qvalue_pair_to_header_element(pair=tuple_) for tuple_ in self.parsed))

    def _add_instance_and_non_accept_language_type(self, instance, other, instance_on_the_right=False):
        if not other:
            return self.__class__(header_value=instance.header_value)
        other_header_value = self._python_value_to_header_str(value=other)
        try:
            self.parse(value=other_header_value)
        except ValueError:
            return self.__class__(header_value=instance.header_value)
        new_header_value = other_header_value + ', ' + instance.header_value if instance_on_the_right else instance.header_value + ', ' + other_header_value
        return self.__class__(header_value=new_header_value)

    def _old_match(self, mask, item):
        """
        Return whether a language tag matches a language range.

        .. warning::

           This is maintained for backward compatibility, and will be
           deprecated in the future.

        This method was WebOb's old criterion for deciding whether a language
        tag matches a language range, used in

        - :meth:`AcceptLanguageValidHeader.__contains__`
        - :meth:`AcceptLanguageValidHeader.best_match`
        - :meth:`AcceptLanguageValidHeader.quality`

        It does not conform to :rfc:`RFC 7231, section 5.3.5
        <7231#section-5.3.5>`, or any of the matching schemes suggested there.

        :param mask: (``str``)

                     | language range

        :param item: (``str``)

                     | language tag. Subtags in language tags are separated by
                       ``-`` (hyphen). If there are underscores (``_``) in this
                       argument, they will be converted to hyphens before
                       checking the match.

        :return: (``bool``) whether the tag in `item` matches the range in
                 `mask`.

        `mask` and `item` are a match if:

        - ``mask == *``.
        - ``mask == item``.
        - If the first subtag of `item` equals `mask`, or if the first subtag
          of `mask` equals `item`.
          This means that::

              >>> instance._old_match(mask='en-gb', item='en')
              True
              >>> instance._old_match(mask='en', item='en-gb')
              True

          Which is different from any of the matching schemes suggested in
          :rfc:`RFC 7231, section 5.3.5 <7231#section-5.3.5>`, in that none of
          those schemes match both more *and* less specific tags.

          However, this method appears to be only designed for language tags
          and ranges with at most two subtags. So with an `item`/language tag
          with more than two subtags like ``zh-Hans-CN``::

              >>> instance._old_match(mask='zh', item='zh-Hans-CN')
              True
              >>> instance._old_match(mask='zh-Hans', item='zh-Hans-CN')
              False

          From commit history, this does not appear to have been from a
          decision to match only the first subtag, but rather because only
          language ranges and tags with at most two subtags were expected.
        """
        item = item.replace('_', '-').lower()
        mask = mask.lower()
        return mask == '*' or item == mask or item.split('-')[0] == mask or (item == mask.split('-')[0])

    def basic_filtering(self, language_tags):
        """
        Return the tags that match the header, using Basic Filtering.

        This is an implementation of the Basic Filtering matching scheme,
        suggested as a matching scheme for the ``Accept-Language`` header in
        :rfc:`RFC 7231, section 5.3.5 <7231#section-5.3.5>`, and defined in
        :rfc:`RFC 4647, section 3.3.1 <4647#section-3.3.1>`. It filters the
        tags in the `language_tags` argument and returns the ones that match
        the header according to the matching scheme.

        :param language_tags: (``iterable``) language tags
        :return: A list of tuples of the form (language tag, qvalue), in
                 descending order of qvalue. If two or more tags have the same
                 qvalue, they are returned in the same order as that in the
                 header of the ranges they matched. If the matched range is the
                 same for two or more tags (i.e. their matched ranges have the
                 same qvalue and the same position in the header), then they
                 are returned in the same order as that in the `language_tags`
                 argument. If `language_tags` is unordered, e.g. if it is a set
                 or a dict, then that order may not be reliable.

        For each tag in `language_tags`:

        1. If the tag matches a non-``*`` language range in the header with
           ``q=0`` (meaning "not acceptable", see :rfc:`RFC 7231, section 5.3.1
           <7231#section-5.3.1>`), the tag is filtered out.
        2. The non-``*`` language ranges in the header that do not have ``q=0``
           are considered in descending order of qvalue; where two or more
           language ranges have the same qvalue, they are considered in the
           order in which they appear in the header.
        3. A language range 'matches a particular language tag if, in a
           case-insensitive comparison, it exactly equals the tag, or if it
           exactly equals a prefix of the tag such that the first character
           following the prefix is "-".' (:rfc:`RFC 4647, section 3.3.1
           <4647#section-3.3.1>`)
        4. If the tag does not match any of the non-``*`` language ranges, and
           there is a ``*`` language range in the header, then if the ``*``
           language range has ``q=0``, the language tag is filtered out,
           otherwise the tag is considered a match.

        (If a range (``*`` or non-``*``) appears in the header more than once
        -- this would not make sense, but is nonetheless a valid header
        according to the RFC -- the first in the header is used for matching,
        and the others are ignored.)
        """
        lowercased_parsed = [(range_.lower(), qvalue) for range_, qvalue in self.parsed]
        lowercased_tags = [tag.lower() for tag in language_tags]
        not_acceptable_ranges = set()
        acceptable_ranges = dict()
        asterisk_qvalue = None
        for position_in_header, (range_, qvalue) in enumerate(lowercased_parsed):
            if range_ == '*':
                if asterisk_qvalue is None:
                    asterisk_qvalue = qvalue
                    asterisk_position = position_in_header
            elif range_ not in acceptable_ranges and range_ not in not_acceptable_ranges:
                if qvalue == 0.0:
                    not_acceptable_ranges.add(range_)
                else:
                    acceptable_ranges[range_] = (qvalue, position_in_header)
        acceptable_ranges = [(range_, qvalue, position_in_header) for range_, (qvalue, position_in_header) in acceptable_ranges.items()]
        acceptable_ranges.sort(key=lambda tuple_: tuple_[2])
        acceptable_ranges.sort(key=lambda tuple_: tuple_[1], reverse=True)

        def match(tag, range_):
            return tag == range_ or tag.startswith(range_ + '-')
        filtered_tags = []
        for index, tag in enumerate(lowercased_tags):
            if any((match(tag=tag, range_=range_) for range_ in not_acceptable_ranges)):
                continue
            matched_range_qvalue = None
            for range_, qvalue, position_in_header in acceptable_ranges:
                if match(tag=tag, range_=range_):
                    matched_range_qvalue = qvalue
                    matched_range_position = position_in_header
                    break
            else:
                if asterisk_qvalue:
                    matched_range_qvalue = asterisk_qvalue
                    matched_range_position = asterisk_position
            if matched_range_qvalue is not None:
                filtered_tags.append((language_tags[index], matched_range_qvalue, matched_range_position))
        filtered_tags.sort(key=lambda tuple_: tuple_[2])
        filtered_tags.sort(key=lambda tuple_: tuple_[1], reverse=True)
        return [(item[0], item[1]) for item in filtered_tags]

    def best_match(self, offers, default_match=None):
        """
        Return the best match from the sequence of language tag `offers`.

        .. warning::

           This is currently maintained for backward compatibility, and will be
           deprecated in the future.

           :meth:`AcceptLanguageValidHeader.best_match` uses its own algorithm
           (one not specified in :rfc:`RFC 7231 <7231>`) to determine what is a
           best match. The algorithm has many issues, and does not conform to
           :rfc:`RFC 7231 <7231>`.

           :meth:`AcceptLanguageValidHeader.lookup` is a possible alternative
           for finding a best match -- it conforms to, and is suggested as a
           matching scheme for the ``Accept-Language`` header in, :rfc:`RFC
           7231, section 5.3.5 <7231#section-5.3.5>` -- but please be aware
           that there are differences in how it determines what is a best
           match. If that is not suitable for the needs of your application,
           you may need to write your own matching using
           :attr:`AcceptLanguageValidHeader.parsed`.

        Each language tag in `offers` is checked against each non-0 range in
        the header. If the two are a match according to WebOb's old criterion
        for a match, the quality value of the match is the qvalue of the
        language range from the header multiplied by the server quality value
        of the offer (if the server quality value is not supplied, it is 1).

        The offer in the match with the highest quality value is the best
        match. If there is more than one match with the highest qvalue, the
        match where the language range has a lower number of '*'s is the best
        match. If the two have the same number of '*'s, the one that shows up
        first in `offers` is the best match.

        :param offers: (iterable)

                       | Each item in the iterable may be a ``str`` language
                         tag, or a (language tag, server quality value)
                         ``tuple`` or ``list``. (The two may be mixed in the
                         iterable.)

        :param default_match: (optional, any type) the value to be returned if
                              there is no match

        :return: (``str``, or the type of `default_match`)

                 | The language tag that is the best match. If there is no
                   match, the value of `default_match` is returned.


        **Issues**:

        - Incorrect tiebreaking when quality values of two matches are the same
          (https://github.com/Pylons/webob/issues/256)::

              >>> header = AcceptLanguageValidHeader(
              ...     header_value='en-gb;q=1, en;q=0.8'
              ... )
              >>> header.best_match(offers=['en', 'en-GB'])
              'en'
              >>> header.best_match(offers=['en-GB', 'en'])
              'en-GB'

              >>> header = AcceptLanguageValidHeader(header_value='en-gb, en')
              >>> header.best_match(offers=['en', 'en-gb'])
              'en'
              >>> header.best_match(offers=['en-gb', 'en'])
              'en-gb'

        - Incorrect handling of ``q=0``::

              >>> header = AcceptLanguageValidHeader(header_value='en;q=0, *')
              >>> header.best_match(offers=['en'])
              'en'

              >>> header = AcceptLanguageValidHeader(header_value='fr, en;q=0')
              >>> header.best_match(offers=['en-gb'], default_match='en')
              'en'

        - Matching only takes into account the first subtag when matching a
          range with more specific or less specific tags::

              >>> header = AcceptLanguageValidHeader(header_value='zh')
              >>> header.best_match(offers=['zh-Hans-CN'])
              'zh-Hans-CN'
              >>> header = AcceptLanguageValidHeader(header_value='zh-Hans')
              >>> header.best_match(offers=['zh-Hans-CN'])
              >>> header.best_match(offers=['zh-Hans-CN']) is None
              True

              >>> header = AcceptLanguageValidHeader(header_value='zh-Hans-CN')
              >>> header.best_match(offers=['zh'])
              'zh'
              >>> header.best_match(offers=['zh-Hans'])
              >>> header.best_match(offers=['zh-Hans']) is None
              True

        """
        warnings.warn('The behavior of AcceptLanguageValidHeader.best_match is currently being maintained for backward compatibility, but it will be deprecated in the future as it does not conform to the RFC.', DeprecationWarning)
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

    def lookup(self, language_tags, default_range=None, default_tag=None, default=None):
        """
        Return the language tag that best matches the header, using Lookup.

        This is an implementation of the Lookup matching scheme,
        suggested as a matching scheme for the ``Accept-Language`` header in
        :rfc:`RFC 7231, section 5.3.5 <7231#section-5.3.5>`, and described in
        :rfc:`RFC 4647, section 3.4 <4647#section-3.4>`.

        Each language range in the header is considered in turn, by descending
        order of qvalue; where qvalues are tied, ranges are considered from
        left to right.

        Each language range in the header represents the most specific tag that
        is an acceptable match: Lookup progressively truncates subtags from the
        end of the range until a matching language tag is found. An example is
        given in :rfc:`RFC 4647, section 3.4 <4647#section-3.4>`, under
        "Example of a Lookup Fallback Pattern":

        ::

            Range to match: zh-Hant-CN-x-private1-private2
            1. zh-Hant-CN-x-private1-private2
            2. zh-Hant-CN-x-private1
            3. zh-Hant-CN
            4. zh-Hant
            5. zh
            6. (default)

        :param language_tags: (``iterable``) language tags

        :param default_range: (optional, ``None`` or ``str``)

                              | If Lookup finds no match using the ranges in
                                the header, and this argument is not None,
                                Lookup will next attempt to match the range in
                                this argument, using the same subtag
                                truncation.

                              | `default_range` cannot be '*', as '*' is
                                skipped in Lookup. See :ref:`note
                                <acceptparse-lookup-asterisk-note>`.

                              | This parameter corresponds to the functionality
                                described in :rfc:`RFC 4647, section 3.4.1
                                <4647#section-3.4.1>`, in the paragraph
                                starting with "One common way to provide for a
                                default is to allow a specific language range
                                to be set as the default..."

        :param default_tag: (optional, ``None`` or ``str``)

                            | At least one of `default_tag` or `default` must
                              be supplied as an argument to the method, to
                              define the defaulting behaviour.

                            | If Lookup finds no match using the ranges in the
                              header and `default_range`, this argument is not
                              ``None``, and it does not match any range in the
                              header with ``q=0`` (exactly, with no subtag
                              truncation), then this value is returned.

                            | This parameter corresponds to "return a
                              particular language tag designated for the
                              operation", one of the examples of "defaulting
                              behavior" described in :rfc:`RFC 4647, section
                              3.4.1 <4647#section-3.4.1>`.

        :param default: (optional, ``None`` or any type, including a callable)

                        | At least one of `default_tag` or `default` must be
                          supplied as an argument to the method, to define the
                          defaulting behaviour.

                        | If Lookup finds no match using the ranges in the
                          header and `default_range`, and `default_tag` is
                          ``None`` or not acceptable because it matches a
                          ``q=0`` range in the header, then Lookup will next
                          examine the `default` argument.

                        | If `default` is a callable, it will be called, and
                          the callable's return value will be returned.

                        | If `default` is not a callable, the value itself will
                          be returned.

                        | The difference between supplying a ``str`` to
                          `default_tag` and `default` is that `default_tag` is
                          checked against ``q=0`` ranges in the header to see
                          if it matches one of the ranges specified as not
                          acceptable, whereas a ``str`` for the `default`
                          argument is simply returned.

                        | This parameter corresponds to the "defaulting
                          behavior" described in :rfc:`RFC 4647, section 3.4.1
                          <4647#section-3.4.1>`

        :return: (``str``, ``None``, or any type)

                 | The best match according to the Lookup matching scheme, or a
                   return value from one of the default arguments.

        **Notes**:

        .. _acceptparse-lookup-asterisk-note:

        - Lookup's behaviour with '*' language ranges in the header may be
          surprising. From :rfc:`RFC 4647, section 3.4 <4647#section-3.4>`:

              In the lookup scheme, this range does not convey enough
              information by itself to determine which language tag is most
              appropriate, since it matches everything.  If the language range
              "*" is followed by other language ranges, it is skipped.  If the
              language range "*" is the only one in the language priority list
              or if no other language range follows, the default value is
              computed and returned.

          So

          ::

              >>> header = AcceptLanguageValidHeader('de, zh, *')
              >>> header.lookup(language_tags=['ja', 'en'], default='default')
              'default'

        - Any tags in `language_tags` and `default_tag` and any tag matched
          during the subtag truncation search for `default_range`, that are an
          exact match for a non-``*`` range with ``q=0`` in the header, are
          considered not acceptable and ruled out.

        - If there is a ``*;q=0`` in the header, then `default_range` and
          `default_tag` have no effect, as ``*;q=0`` means that all languages
          not already matched by other ranges within the header are
          unacceptable.
        """
        if default_tag is None and default is None:
            raise TypeError('`default_tag` and `default` arguments cannot both be None.')
        if default_range == '*':
            raise ValueError('default_range cannot be *.')
        parsed = list(self.parsed)
        tags = language_tags
        not_acceptable_ranges = []
        acceptable_ranges = []
        asterisk_non0_found = False
        asterisk_q0_found = False
        for range_, qvalue in parsed:
            if qvalue == 0.0:
                if range_ == '*':
                    asterisk_q0_found = True
                else:
                    not_acceptable_ranges.append(range_.lower())
            elif not asterisk_q0_found and range_ == '*':
                asterisk_non0_found = True
            else:
                acceptable_ranges.append((range_, qvalue))
        acceptable_ranges.sort(key=lambda tuple_: tuple_[1], reverse=True)
        acceptable_ranges = [tuple_[0] for tuple_ in acceptable_ranges]
        lowered_tags = [tag.lower() for tag in tags]

        def best_match(range_):
            subtags = range_.split('-')
            while True:
                for index, tag in enumerate(lowered_tags):
                    if tag in not_acceptable_ranges:
                        continue
                    if tag == range_:
                        return tags[index]
                try:
                    subtag_before_this = subtags[-2]
                except IndexError:
                    break
                if len(subtag_before_this) == 1 and (subtag_before_this.isdigit() or subtag_before_this.isalpha()):
                    subtags.pop(-1)
                subtags.pop(-1)
                range_ = '-'.join(subtags)
        for range_ in acceptable_ranges:
            match = best_match(range_=range_.lower())
            if match is not None:
                return match
        if not asterisk_q0_found:
            if default_range is not None:
                lowered_default_range = default_range.lower()
                match = best_match(range_=lowered_default_range)
                if match is not None:
                    return match
            if default_tag is not None:
                lowered_default_tag = default_tag.lower()
                if lowered_default_tag not in not_acceptable_ranges:
                    return default_tag
        try:
            return default()
        except TypeError:
            return default

    def quality(self, offer):
        """
        Return quality value of given offer, or ``None`` if there is no match.

        .. warning::

           This is currently maintained for backward compatibility, and will be
           deprecated in the future.

           :meth:`AcceptLanguageValidHeader.quality` uses its own algorithm
           (one not specified in :rfc:`RFC 7231 <7231>`) to determine what is a
           best match. The algorithm has many issues, and does not conform to
           :rfc:`RFC 7231 <7231>`.

           What should be considered a match depends on the needs of your
           application (for example, should a language range in the header
           match a more specific language tag offer, or a less specific tag
           offer?) :rfc:`RFC 7231, section 5.3.5 <7231#section-5.3.5>` suggests
           three matching schemes from :rfc:`RFC 4647 <4647>`, two of which
           WebOb supports with
           :meth:`AcceptLanguageValidHeader.basic_filtering` and
           :meth:`AcceptLanguageValidHeader.lookup` (we interpret the RFC to
           mean that Extended Filtering cannot apply for the
           ``Accept-Language`` header, as the header only accepts basic
           language ranges.) :meth:`AcceptLanguageValidHeader.basic_filtering`
           returns quality values with the matched language tags.
           :meth:`AcceptLanguageValidHeader.lookup` returns a language tag
           without the quality value, but the quality value is less likely to
           be useful when we are looking for a best match.

           If these are not suitable or sufficient for the needs of your
           application, you may need to write your own matching using
           :attr:`AcceptLanguageValidHeader.parsed`.

        :param offer: (``str``) language tag offer
        :return: (``float`` or ``None``)

                 | The highest quality value from the language range(s) that
                   match the `offer`, or ``None`` if there is no match.


        **Issues**:

        - Incorrect handling of ``q=0`` and ``*``::

              >>> header = AcceptLanguageValidHeader(header_value='en;q=0, *')
              >>> header.quality(offer='en')
              1.0

        - Matching only takes into account the first subtag when matching a
          range with more specific or less specific tags::

              >>> header = AcceptLanguageValidHeader(header_value='zh')
              >>> header.quality(offer='zh-Hans-CN')
              1.0
              >>> header = AcceptLanguageValidHeader(header_value='zh-Hans')
              >>> header.quality(offer='zh-Hans-CN')
              >>> header.quality(offer='zh-Hans-CN') is None
              True

              >>> header = AcceptLanguageValidHeader(header_value='zh-Hans-CN')
              >>> header.quality(offer='zh')
              1.0
              >>> header.quality(offer='zh-Hans')
              >>> header.quality(offer='zh-Hans') is None
              True

        """
        warnings.warn('The behavior of AcceptLanguageValidHeader.quality iscurrently being maintained for backward compatibility, but it will be deprecated in the future as it does not conform to the RFC.', DeprecationWarning)
        bestq = 0
        for mask, q in self.parsed:
            if self._old_match(mask, offer):
                bestq = max(bestq, q)
        return bestq or None