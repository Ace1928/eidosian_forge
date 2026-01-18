from __future__ import annotations
from collections import namedtuple

    :param tag: Integer tag number
    :param group: Which :py:data:`~PIL.TiffTags.TAGS_V2_GROUPS` to look in

    .. versionadded:: 8.3.0

    :returns: Taginfo namedtuple, From the ``TAGS_V2`` info if possible,
        otherwise just populating the value and name from ``TAGS``.
        If the tag is not recognized, "unknown" is returned for the name

    