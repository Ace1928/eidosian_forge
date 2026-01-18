from __future__ import annotations
import typing as t
from ..http import parse_list_header
def _get_real_value(self, trusted: int, value: str | None) -> str | None:
    """Get the real value from a list header based on the configured
        number of trusted proxies.

        :param trusted: Number of values to trust in the header.
        :param value: Comma separated list header value to parse.
        :return: The real value, or ``None`` if there are fewer values
            than the number of trusted proxies.

        .. versionchanged:: 1.0
            Renamed from ``_get_trusted_comma``.

        .. versionadded:: 0.15
        """
    if not (trusted and value):
        return None
    values = parse_list_header(value)
    if len(values) >= trusted:
        return values[-trusted]
    return None