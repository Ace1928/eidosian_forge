from __future__ import annotations
import typing
from collections import OrderedDict
from enum import Enum, auto
from threading import RLock
def _prepare_for_method_change(self) -> Self:
    """
        Remove content-specific header fields before changing the request
        method to GET or HEAD according to RFC 9110, Section 15.4.
        """
    content_specific_headers = ['Content-Encoding', 'Content-Language', 'Content-Location', 'Content-Type', 'Content-Length', 'Digest', 'Last-Modified']
    for header in content_specific_headers:
        self.discard(header)
    return self