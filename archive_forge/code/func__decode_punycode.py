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
def _decode_punycode(label: str) -> str:
    lowered = label.lower()
    looks_like_puny = lowered.startswith('xn--')
    if looks_like_puny:
        try:
            return idna.decode(lowered)
        except (UnicodeError, IndexError):
            pass
    return lowered