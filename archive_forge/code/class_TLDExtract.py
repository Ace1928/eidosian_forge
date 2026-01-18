from __future__ import annotations
import logging
import os
import urllib.parse
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from functools import wraps
import idna
import requests
from .cache import DiskCache, get_cache_dir
from .remote import lenient_netloc, looks_like_ip, looks_like_ipv6
from .suffix_list import get_suffix_lists
class TLDExtract:
    """A callable for extracting, subdomain, domain, and suffix components from a URL."""

    def __init__(self, cache_dir: str | None=get_cache_dir(), suffix_list_urls: Sequence[str]=PUBLIC_SUFFIX_LIST_URLS, fallback_to_snapshot: bool=True, include_psl_private_domains: bool=False, extra_suffixes: Sequence[str]=(), cache_fetch_timeout: str | float | None=CACHE_TIMEOUT) -> None:
        """Construct a callable for extracting subdomain, domain, and suffix components from a URL.

        Upon calling it, it first checks for a JSON in `cache_dir`. By default,
        the `cache_dir` will live in the tldextract directory. You can disable
        the caching functionality of this module by setting `cache_dir` to `None`.

        If the cached version does not exist (such as on the first run), HTTP request the URLs in
        `suffix_list_urls` in order, until one returns public suffix list data. To disable HTTP
        requests, set this to an empty sequence.

        The default list of URLs point to the latest version of the Mozilla Public Suffix List and
        its mirror, but any similar document could be specified. Local files can be specified by
        using the `file://` protocol. (See `urllib2` documentation.)

        If there is no cached version loaded and no data is found from the `suffix_list_urls`,
        the module will fall back to the included TLD set snapshot. If you do not want
        this behavior, you may set `fallback_to_snapshot` to False, and an exception will be
        raised instead.

        The Public Suffix List includes a list of "private domains" as TLDs,
        such as blogspot.com. These do not fit `tldextract`'s definition of a
        suffix, so these domains are excluded by default. If you'd like them
        included instead, set `include_psl_private_domains` to True.

        You can pass additional suffixes in `extra_suffixes` argument without changing list URL

        cache_fetch_timeout is passed unmodified to the underlying request object
        per the requests documentation here:
        http://docs.python-requests.org/en/master/user/advanced/#timeouts

        cache_fetch_timeout can also be set to a single value with the
        environment variable TLDEXTRACT_CACHE_TIMEOUT, like so:

        TLDEXTRACT_CACHE_TIMEOUT="1.2"

        When set this way, the same timeout value will be used for both connect
        and read timeouts
        """
        suffix_list_urls = suffix_list_urls or ()
        self.suffix_list_urls = tuple((url.strip() for url in suffix_list_urls if url.strip()))
        self.fallback_to_snapshot = fallback_to_snapshot
        if not (self.suffix_list_urls or cache_dir or self.fallback_to_snapshot):
            raise ValueError('The arguments you have provided disable all ways for tldextract to obtain data. Please provide a suffix list data, a cache_dir, or set `fallback_to_snapshot` to `True`.')
        self.include_psl_private_domains = include_psl_private_domains
        self.extra_suffixes = extra_suffixes
        self._extractor: _PublicSuffixListTLDExtractor | None = None
        self.cache_fetch_timeout = float(cache_fetch_timeout) if isinstance(cache_fetch_timeout, str) else cache_fetch_timeout
        self._cache = DiskCache(cache_dir)

    def __call__(self, url: str, include_psl_private_domains: bool | None=None, session: requests.Session | None=None) -> ExtractResult:
        """Alias for `extract_str`."""
        return self.extract_str(url, include_psl_private_domains, session=session)

    def extract_str(self, url: str, include_psl_private_domains: bool | None=None, session: requests.Session | None=None) -> ExtractResult:
        """Take a string URL and splits it into its subdomain, domain, and suffix components.

        I.e. its effective TLD, gTLD, ccTLD, etc. components.

        >>> extractor = TLDExtract()
        >>> extractor.extract_str('http://forums.news.cnn.com/')
        ExtractResult(subdomain='forums.news', domain='cnn', suffix='com', is_private=False)
        >>> extractor.extract_str('http://forums.bbc.co.uk/')
        ExtractResult(subdomain='forums', domain='bbc', suffix='co.uk', is_private=False)

        Allows configuring the HTTP request via the optional `session`
        parameter. For example, if you need to use a HTTP proxy. See also
        `requests.Session`.

        >>> import requests
        >>> session = requests.Session()
        >>> # customize your session here
        >>> with session:
        ...     extractor.extract_str("http://forums.news.cnn.com/", session=session)
        ExtractResult(subdomain='forums.news', domain='cnn', suffix='com', is_private=False)
        """
        return self._extract_netloc(lenient_netloc(url), include_psl_private_domains, session=session)

    def extract_urllib(self, url: urllib.parse.ParseResult | urllib.parse.SplitResult, include_psl_private_domains: bool | None=None, session: requests.Session | None=None) -> ExtractResult:
        """Take the output of urllib.parse URL parsing methods and further splits the parsed URL.

        Splits the parsed URL into its subdomain, domain, and suffix
        components, i.e. its effective TLD, gTLD, ccTLD, etc. components.

        This method is like `extract_str` but faster, as the string's domain
        name has already been parsed.

        >>> extractor = TLDExtract()
        >>> extractor.extract_urllib(urllib.parse.urlsplit('http://forums.news.cnn.com/'))
        ExtractResult(subdomain='forums.news', domain='cnn', suffix='com', is_private=False)
        >>> extractor.extract_urllib(urllib.parse.urlsplit('http://forums.bbc.co.uk/'))
        ExtractResult(subdomain='forums', domain='bbc', suffix='co.uk', is_private=False)
        """
        return self._extract_netloc(url.netloc, include_psl_private_domains, session=session)

    def _extract_netloc(self, netloc: str, include_psl_private_domains: bool | None, session: requests.Session | None=None) -> ExtractResult:
        netloc_with_ascii_dots = netloc.replace('。', '.').replace('．', '.').replace('｡', '.')
        min_num_ipv6_chars = 4
        if len(netloc_with_ascii_dots) >= min_num_ipv6_chars and netloc_with_ascii_dots[0] == '[' and (netloc_with_ascii_dots[-1] == ']'):
            if looks_like_ipv6(netloc_with_ascii_dots[1:-1]):
                return ExtractResult('', netloc_with_ascii_dots, '', is_private=False)
        labels = netloc_with_ascii_dots.split('.')
        suffix_index, is_private = self._get_tld_extractor(session=session).suffix_index(labels, include_psl_private_domains=include_psl_private_domains)
        num_ipv4_labels = 4
        if suffix_index == len(labels) == num_ipv4_labels and looks_like_ip(netloc_with_ascii_dots):
            return ExtractResult('', netloc_with_ascii_dots, '', is_private)
        suffix = '.'.join(labels[suffix_index:]) if suffix_index != len(labels) else ''
        subdomain = '.'.join(labels[:suffix_index - 1]) if suffix_index >= 2 else ''
        domain = labels[suffix_index - 1] if suffix_index else ''
        return ExtractResult(subdomain, domain, suffix, is_private)

    def update(self, fetch_now: bool=False, session: requests.Session | None=None) -> None:
        """Force fetch the latest suffix list definitions."""
        self._extractor = None
        self._cache.clear()
        if fetch_now:
            self._get_tld_extractor(session=session)

    @property
    def tlds(self, session: requests.Session | None=None) -> list[str]:
        """Returns the list of tld's used by default.

        This will vary based on `include_psl_private_domains` and `extra_suffixes`
        """
        return list(self._get_tld_extractor(session=session).tlds())

    def _get_tld_extractor(self, session: requests.Session | None=None) -> _PublicSuffixListTLDExtractor:
        """Get or compute this object's TLDExtractor.

        Looks up the TLDExtractor in roughly the following order, based on the
        settings passed to __init__:

        1. Memoized on `self`
        2. Local system _cache file
        3. Remote PSL, over HTTP
        4. Bundled PSL snapshot file
        """
        if self._extractor:
            return self._extractor
        public_tlds, private_tlds = get_suffix_lists(cache=self._cache, urls=self.suffix_list_urls, cache_fetch_timeout=self.cache_fetch_timeout, fallback_to_snapshot=self.fallback_to_snapshot, session=session)
        if not any([public_tlds, private_tlds, self.extra_suffixes]):
            raise ValueError('No tlds set. Cannot proceed without tlds.')
        self._extractor = _PublicSuffixListTLDExtractor(public_tlds=public_tlds, private_tlds=private_tlds, extra_tlds=list(self.extra_suffixes), include_psl_private_domains=self.include_psl_private_domains)
        return self._extractor