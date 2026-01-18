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
@dataclass(order=True)
class ExtractResult:
    """A URL's extracted subdomain, domain, and suffix.

    Also contains metadata, like a flag that indicates if the URL has a private suffix.
    """
    subdomain: str
    domain: str
    suffix: str
    is_private: bool

    @property
    def registered_domain(self) -> str:
        """Joins the domain and suffix fields with a dot, if they're both set.

        >>> extract('http://forums.bbc.co.uk').registered_domain
        'bbc.co.uk'
        >>> extract('http://localhost:8080').registered_domain
        ''
        """
        if self.suffix and self.domain:
            return f'{self.domain}.{self.suffix}'
        return ''

    @property
    def fqdn(self) -> str:
        """Returns a Fully Qualified Domain Name, if there is a proper domain/suffix.

        >>> extract('http://forums.bbc.co.uk/path/to/file').fqdn
        'forums.bbc.co.uk'
        >>> extract('http://localhost:8080').fqdn
        ''
        """
        if self.suffix and (self.domain or self.is_private):
            return '.'.join((i for i in (self.subdomain, self.domain, self.suffix) if i))
        return ''

    @property
    def ipv4(self) -> str:
        """Returns the ipv4 if that is what the presented domain/url is.

        >>> extract('http://127.0.0.1/path/to/file').ipv4
        '127.0.0.1'
        >>> extract('http://127.0.0.1.1/path/to/file').ipv4
        ''
        >>> extract('http://256.1.1.1').ipv4
        ''
        """
        if self.domain and (not (self.suffix or self.subdomain)) and looks_like_ip(self.domain):
            return self.domain
        return ''

    @property
    def ipv6(self) -> str:
        """Returns the ipv6 if that is what the presented domain/url is.

        >>> extract('http://[aBcD:ef01:2345:6789:aBcD:ef01:127.0.0.1]/path/to/file').ipv6
        'aBcD:ef01:2345:6789:aBcD:ef01:127.0.0.1'
        >>> extract('http://[aBcD:ef01:2345:6789:aBcD:ef01:127.0.0.1.1]/path/to/file').ipv6
        ''
        >>> extract('http://[aBcD:ef01:2345:6789:aBcD:ef01:256.0.0.1]').ipv6
        ''
        """
        min_num_ipv6_chars = 4
        if len(self.domain) >= min_num_ipv6_chars and self.domain[0] == '[' and (self.domain[-1] == ']') and (not (self.suffix or self.subdomain)):
            debracketed = self.domain[1:-1]
            if looks_like_ipv6(debracketed):
                return debracketed
        return ''