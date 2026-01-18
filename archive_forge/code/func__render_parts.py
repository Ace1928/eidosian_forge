from __future__ import absolute_import
import email.utils
import mimetypes
import re
from .packages import six
def _render_parts(self, header_parts):
    """
        Helper function to format and quote a single header.

        Useful for single headers that are composed of multiple items. E.g.,
        'Content-Disposition' fields.

        :param header_parts:
            A sequence of (k, v) tuples or a :class:`dict` of (k, v) to format
            as `k1="v1"; k2="v2"; ...`.
        """
    parts = []
    iterable = header_parts
    if isinstance(header_parts, dict):
        iterable = header_parts.items()
    for name, value in iterable:
        if value is not None:
            parts.append(self._render_part(name, value))
    return u'; '.join(parts)