from __future__ import unicode_literals
import functools
import re
from datetime import timedelta
import logging
import io
def _check_contiguity(srt, expected_start, actual_start, warn_only):
    """
    If ``warn_only`` is False, raise :py:class:`SRTParseError` with diagnostic
    info if expected_start does not equal actual_start. Otherwise, log a
    warning.

    :param str srt: The data being matched
    :param int expected_start: The expected next start, as from the last
                               iteration's match.end()
    :param int actual_start: The actual start, as from this iteration's
                             match.start()
    :raises SRTParseError: If the matches are not contiguous and ``warn_only``
                           is False
    """
    if expected_start != actual_start:
        unmatched_content = srt[expected_start:actual_start]
        if expected_start == 0 and (unmatched_content.isspace() or unmatched_content == '\ufeff'):
            return
        if warn_only:
            LOG.warning('Skipped unparseable SRT data: %r', unmatched_content)
        else:
            raise SRTParseError(expected_start, actual_start, unmatched_content)