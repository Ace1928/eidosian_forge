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